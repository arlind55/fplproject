"""fplvalue: load the fplproject data snapshots, project expected points, pick captains and squads.

    import fplvalue as fv
    snap = fv.load()                       # data/current
    last = fv.last_season(snap)            # previous season's final snapshot (prior)
    proj = fv.project_snapshot(snap, last) # one row per player, gw_<n> columns
"""
from .load import load, last_season, snapshot_before, Snapshot, POS          # noqa: F401
from .project import Params, State, project, project_snapshot, state_from_snapshot, state_as_of, COMPONENTS  # noqa: F401
