"""Daily team-news scan: selection risks the FPL flags don't capture, written to data/overrides.csv.

FPL's status / chance of playing covers injuries and suspensions. It says nothing when a player is dropped
after a fallout with the manager, rested for a cup tie, or losing his place to a new signing. This asks
Claude (with web search) to read the latest press-conference and team-news reports for the most-owned and
most expensive players, and returns a start probability for each player with a credible story. Those
rows replace the auto rows in data/overrides.csv (rows you wrote by hand, origin=manual, are never
touched and win over an auto row for the same player), then the projections are rewritten.

    python scripts/news_scan.py                    # skips once the next deadline has passed
    python scripts/news_scan.py --force            # run anyway
    python scripts/news_scan.py --republish        # no scan: just re-apply data/overrides.csv
    python scripts/news_scan.py --summary out.md   # also write what changed, for the notification

Needs ANTHROPIC_API_KEY (does nothing without it). NEWS_MODEL picks the model (default claude-opus-5-5).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from fplvalue import overrides  # noqa: E402
from refresh_flags import CURRENT_DIR, republish  # noqa: E402

MODEL = os.environ.get('NEWS_MODEL', 'claude-opus-5-5')
MAX_SEARCHES = 20
HORIZON = 4                      # an auto row can reach at most this many gameweeks ahead
P_MIN, P_MAX = 0.05, 0.95        # an auto row never rules a player fully in or out

SYSTEM = """You are the team-news desk for a Fantasy Premier League projection model.

The model already knows each player's recent starts and FPL's own injury flags (status, chance of playing,
and the FPL news text you are given). Your job is the selection news those miss: a player dropped or
frozen out after a falling-out with the manager or club discipline, rotation the manager has signalled
(cup or European ties, fixture congestion), a new signing or a returning first-choice player taking the
place, a manager saying a player is "not ready" for 90 minutes, a change of manager that changes who starts.
Positive news counts too: a player the manager has said will start, or who has won his place back.

Use web search to read this week's press-conference quotes and team-news reports for the listed players'
clubs. Prefer the manager's own words and established outlets over rumour. Ignore injury-only stories that
the FPL news text already covers.

When you are done, call report_adjustments exactly once with the full list of adjustments that should
apply now. You are also given the adjustments you made on previous runs: keep each one that still holds
(updating it if the story moved on), and leave out any that no longer apply. Only include a player when a
specific report makes his chance of starting clearly different from his recent record; an empty list is
a fine answer. p_start is your probability that he starts the gameweek, from 0 to 1."""

TOOL = {
    'name': 'report_adjustments',
    'description': 'Report the start-probability adjustments that should apply now. Call once, at the end.',
    'strict': True,
    'input_schema': {
        'type': 'object',
        'additionalProperties': False,
        'required': ['adjustments'],
        'properties': {'adjustments': {'type': 'array', 'items': {
            'type': 'object',
            'additionalProperties': False,
            'required': ['player_id', 'p_start', 'gw_from', 'gw_to', 'confidence', 'note', 'source_url'],
            'properties': {
                'player_id': {'type': 'integer', 'description': 'FPL id from the player list'},
                'p_start': {'type': 'number', 'description': 'Probability he starts, 0-1'},
                'gw_from': {'type': 'integer', 'description': 'First gameweek this applies to'},
                'gw_to': {'type': 'integer', 'description': 'Last gameweek this applies to'},
                'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']},
                'note': {'type': 'string', 'description': 'One sentence for the site, under 120 characters'},
                'source_url': {'type': 'string', 'description': 'Link to the report it is based on'},
            }}}},
    },
}


def candidates(players: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Players worth checking: owned by 1%+ of managers or priced £6.5m+, and not already ruled out by FPL."""
    p = players.copy()
    p['price'] = p['now_cost'] / 10 if p['now_cost'].max() > 20 else p['now_cost']
    p['club'] = p['team'].map(teams.set_index('id')['name'])
    p['pos'] = p['element_type'].map({1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'})
    keep = ((p['selected_by_percent'] >= 1) | (p['price'] >= 6.5)) & ~p['status'].isin(['u', 'n', 'i', 's'])
    return p[keep].sort_values(['club', 'selected_by_percent'], ascending=[True, False])


def ask(cands: pd.DataFrame, previous: pd.DataFrame, next_gw: int, deadline: str) -> list[dict]:
    import anthropic

    lines = [f"{r.id} | {r.web_name} | {r.club} | {r.pos} | £{r.price:.1f}m | {r.selected_by_percent:.1f}% owned"
             + (f" | FPL news: {r.news}" if isinstance(r.news, str) and r.news else '')
             for r in cands.itertuples()]
    prev = previous[['id', 'web_name', 'p_start', 'gw_from', 'gw_to', 'note', 'source']].to_dict('records')
    user = (f"Today is {datetime.now(timezone.utc):%A %d %B %Y}. The next gameweek is GW{next_gw}, deadline {deadline} UTC.\n\n"
            f"Players (id | name | club | position | price | ownership | FPL news):\n" + '\n'.join(lines)
            + f"\n\nYour adjustments from previous runs:\n{json.dumps(prev, default=str) if prev else 'none'}")

    client = anthropic.Anthropic()
    messages = [{'role': 'user', 'content': user}]
    for _ in range(8):     # pause_turn continuations and nudges
        r = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, messages=messages,
            tools=[{'type': 'web_search_20260209', 'name': 'web_search', 'max_uses': MAX_SEARCHES}, TOOL],
            output_config={'effort': 'medium'},
            betas=['server-side-fallback-2026-07-01'], fallbacks='default',
        )
        if r.stop_reason == 'refusal':
            raise RuntimeError(f'model declined: {r.stop_details}')
        call = next((b for b in r.content if b.type == 'tool_use' and b.name == TOOL['name']), None)
        if call is not None:
            return call.input['adjustments']
        messages.append({'role': 'assistant', 'content': r.content})
        if r.stop_reason != 'pause_turn':
            messages.append({'role': 'user', 'content': 'Call report_adjustments now with your final list.'})
    raise RuntimeError('no report_adjustments call')


def clean(adj: list[dict], players: pd.DataFrame, manual_ids: set, next_gw: int) -> pd.DataFrame:
    """Guard rails on what the model returns before it can move a projection."""
    names = players.set_index('id')['web_name']
    rows, today = {}, datetime.now(timezone.utc).strftime('%Y-%m-%d')
    for a in adj:
        pid = int(a['player_id'])
        if a['confidence'] == 'low' or pid not in names.index or pid in manual_ids:
            continue
        lo = max(int(a['gw_from']), next_gw)
        hi = min(max(int(a['gw_to']), lo), next_gw + HORIZON - 1)
        rows[pid] = {'id': pid, 'web_name': names[pid], 'p_start': round(min(max(float(a['p_start']), P_MIN), P_MAX), 2),
                     'gw_from': lo, 'gw_to': hi, 'note': ' '.join(str(a['note']).split())[:160],
                     'source': a['source_url'], 'origin': 'auto', 'updated': today}
    return pd.DataFrame(list(rows.values()), columns=overrides.COLUMNS)


def describe(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    o, n = old.set_index('id'), new.set_index('id')
    out = []
    for pid in n.index:
        r = n.loc[pid]
        if pid not in o.index:
            out.append(f"- **{r.web_name}**: start chance set to {r.p_start:.0%} (GW{r.gw_from}–{r.gw_to}). {r.note} [source]({r.source})")
        elif (o.at[pid, 'p_start'], o.at[pid, 'gw_to'], o.at[pid, 'note']) != (r.p_start, r.gw_to, r.note):
            out.append(f"- **{r.web_name}**: {o.at[pid, 'p_start']:.0%} → {r.p_start:.0%} (GW{r.gw_from}–{r.gw_to}). {r.note} [source]({r.source})")
    for pid in o.index.difference(n.index):
        out.append(f"- **{o.at[pid, 'web_name']}**: adjustment removed, back to the model's own estimate")
    return out


def main(args: list[str]) -> None:
    summary = Path(args[args.index('--summary') + 1]) if '--summary' in args else None
    if '--republish' in args:
        republish('news_updated_at')
        return
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print('ANTHROPIC_API_KEY not set, skipping the news scan')
        return
    events = pd.read_csv(CURRENT_DIR / 'events.csv')
    nxt = events.loc[events['is_next'].astype(str).str.lower() == 'true']
    if not len(nxt):
        print('no next gameweek, nothing to do')
        return
    next_gw, deadline = int(nxt['id'].iloc[0]), pd.to_datetime(nxt['deadline_time'].iloc[0], utc=True)
    if deadline < pd.Timestamp.now(tz='UTC') and '--force' not in args:
        print(f'GW{next_gw} deadline has passed, nothing to do until the weekly refresh')
        return

    players = pd.read_csv(CURRENT_DIR / 'players.csv')
    teams = pd.read_csv(CURRENT_DIR / 'teams.csv')
    cur = overrides.read()
    manual, auto = cur[cur['origin'] == 'manual'], cur[cur['origin'] != 'manual']
    auto = auto[auto['gw_to'].isna() | (auto['gw_to'] >= next_gw)]

    adj = ask(candidates(players, teams), auto, next_gw, f'{deadline:%a %d %b %H:%M}')
    new = clean(adj, players, set(manual['id'].astype(int)), next_gw)
    changes = describe(cur[cur['origin'] != 'manual'], new)
    print(f'{len(new)} news adjustments, {len(changes)} changes')
    print('\n'.join(changes))
    if not changes:
        return
    pd.concat([manual, new], ignore_index=True).astype({'id': int, 'gw_from': 'Int64', 'gw_to': 'Int64'}).to_csv(overrides.PATH, index=False)
    republish('news_updated_at')
    if summary:
        summary.write_text(f"GW{next_gw} team news, {datetime.now(timezone.utc):%a %d %b %H:%M} UTC\n\n" + '\n'.join(changes) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
