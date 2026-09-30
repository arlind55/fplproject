"""Kept for backwards compatibility: the team model now lives in the fplvalue package."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fplvalue.team_model import *  # noqa: F401,F403
from fplvalue.team_model import build, team_matches, team_ratings, fixture_model  # noqa: F401
