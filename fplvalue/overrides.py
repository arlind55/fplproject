"""Start-probability overrides from team news the FPL flags don't cover (a manager fallout, a suspension
from club discipline, rotation for a cup tie, a new signing taking the shirt).

    data/overrides.csv   one row per story:
        id        FPL element id
        web_name  for reading the file (not used by the model)
        p_start   chance of starting, 0-1, replacing the model's recent-starts estimate
        gw_from   first gameweek it applies to (blank: from the next one)
        gw_to     last gameweek it applies to (blank: until removed)
        note      one line shown on the site
        source    link to the story
        origin    manual (you) or auto (scripts/news_scan.py); a manual row for the same player wins,
                  and the scanner never rewrites manual rows
        updated   date the row was last written

Applied to live projections only (publish.write), never to the backtest, which has no news to go on.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .load import DATA

PATH = DATA / 'overrides.csv'
COLUMNS = ['id', 'web_name', 'p_start', 'gw_from', 'gw_to', 'note', 'source', 'origin', 'updated']


def read(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or PATH)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    o = pd.read_csv(path, dtype={'note': str, 'source': str, 'origin': str, 'web_name': str}, keep_default_na=False,
                    na_values={c: [''] for c in ('id', 'p_start', 'gw_from', 'gw_to')})
    for c in COLUMNS:
        if c not in o:
            o[c] = np.nan if c in ('gw_from', 'gw_to') else ''
    o['id'] = pd.to_numeric(o['id'], errors='coerce')
    o['p_start'] = pd.to_numeric(o['p_start'], errors='coerce').clip(0, 1)
    o['origin'] = o['origin'].str.strip().str.lower().replace('', 'manual')
    return o[o['id'].notna() & o['p_start'].notna()][COLUMNS].reset_index(drop=True)


def per_gw(o: pd.DataFrame, gws: list[int]) -> pd.DataFrame:
    """One row per (player, gameweek) in gws: p_start_adj and adj_note. Manual rows beat auto rows."""
    rows = []
    for r in o.itertuples(index=False):
        lo = int(r.gw_from) if pd.notna(r.gw_from) else gws[0]
        hi = int(r.gw_to) if pd.notna(r.gw_to) else gws[-1]
        for gw in gws:
            if lo <= gw <= hi:
                span = f'GW{lo}' if lo == hi else f'GW{lo}–{hi}' if pd.notna(r.gw_to) else f'from GW{lo}'
                rows.append((int(r.id), gw, float(r.p_start), f"{r.note or 'Team news'} ({span})", r.origin == 'manual'))
    df = pd.DataFrame(rows, columns=['id', 'event', 'p_start_adj', 'adj_note', 'manual'])
    df = df.sort_values('manual').drop_duplicates(['id', 'event'], keep='last')
    return df.drop(columns='manual')
