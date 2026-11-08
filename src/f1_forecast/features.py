"""Shared historical and pre-race transformations; no current-race outcome features."""

import numpy as np
import pandas as pd

from .data import RACE_KEYS, ROW_KEYS, require_columns, validate_history, validate_identity

V1_FEATURES = [
    "grid_position", "qualifying_position", "driver_avg_finish_last5",
    "driver_avg_points_last5", "driver_win_rate_last5", "driver_prior_starts",
    "team_avg_points_last5", "pit_lane_start",
]
V2_FEATURES = V1_FEATURES + ["q1_gap_s", "q2_gap_s", "q3_gap_s", "has_q2_time", "has_q3_time"]
V3_FEATURES = V2_FEATURES + ["grid_places_lost_vs_qualifying",
                            "grid_places_gained_vs_qualifying", "starts_p14_or_worse"]
FEATURE_SETS = {"V1": V1_FEATURES, "V2": V2_FEATURES, "V3": V3_FEATURES}


def lap_seconds(value):
    if pd.isna(value) or str(value).strip() in ("", "-", "\\N"):
        return np.nan
    parts = str(value).split(":")
    if len(parts) not in (1, 2):
        raise ValueError(f"Invalid qualifying time: {value}")
    seconds = float(parts[-1]) + (60 * int(parts[0]) if len(parts) == 2 else 0)
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Invalid qualifying time: {value}")
    return seconds


def _engineer(frame):
    frame = frame.sort_values(["race_date", *ROW_KEYS]).reset_index(drop=True).copy()
    for source, target in [("finish_position", "driver_avg_finish_last5"),
                           ("points", "driver_avg_points_last5"),
                           ("winner", "driver_win_rate_last5")]:
        frame[target] = frame.groupby("driver_id")[source].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    frame["driver_prior_starts"] = frame.groupby("driver_id").cumcount()
    team_keys = RACE_KEYS + ["race_date", "team_id"]
    teams = frame.groupby(team_keys, as_index=False).points.sum(min_count=1)
    teams = teams.sort_values(["race_date", *RACE_KEYS, "team_id"])
    teams["team_avg_points_last5"] = teams.groupby("team_id").points.transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    frame = frame.merge(teams[team_keys + ["team_avg_points_last5"]], on=team_keys,
                        how="left", validate="many_to_one")
    bad_grid = frame.grid_position.isna() | frame.grid_position.le(0)
    max_grid = frame.grid_position.where(~bad_grid).groupby(
        [frame[k] for k in RACE_KEYS]).transform("max").fillna(0)
    frame["pit_lane_start"] = bad_grid.astype(int)
    frame.loc[bad_grid, "grid_position"] = max_grid[bad_grid] + 1
    frame["qualifying_position"] = frame.qualifying_position.where(frame.qualifying_position > 0)
    displacement = frame.grid_position - frame.qualifying_position
    frame["grid_places_lost_vs_qualifying"] = displacement.clip(lower=0)
    frame["grid_places_gained_vs_qualifying"] = (-displacement).clip(lower=0)
    frame["starts_p14_or_worse"] = frame.grid_position.ge(14).astype(int)
    for segment in ("q1", "q2", "q3"):
        seconds = frame[segment].map(lap_seconds)
        fastest = seconds.groupby([frame[k] for k in RACE_KEYS]).transform("min")
        frame[f"{segment}_gap_s"] = seconds - fastest
        if segment != "q1":
            frame[f"has_{segment}_time"] = seconds.notna().astype(int)
    return frame


def build_features(history):
    return _engineer(validate_history(history))


def validate_grid(grid):
    require_columns(grid, ["grid_position", "qualifying_position", "q1", "q2", "q3"])
    validate_identity(grid)
    grid = grid.copy()
    if len(grid) < 2 or grid.driver_id.duplicated().any():
        raise ValueError("Grid requires at least two unique driver IDs")
    for col in ("grid_position", "qualifying_position"):
        grid[col] = pd.to_numeric(grid[col], errors="raise")
        values = grid[col]
        if not (np.isfinite(values) & values.ge(0) & values.le(len(grid)) & (values % 1 == 0)).all():
            raise ValueError(f"Invalid {col}: use integer positions; 0 denotes pit lane/no position")
        positive = values[values > 0]
        if positive.duplicated().any():
            raise ValueError(f"Duplicate {col} in grid")
    return grid


def build_target_features(history, grid, race_date, season, round_number):
    """Append outcome-free target rows and apply the identical lagged feature formula."""
    grid = validate_grid(grid)
    history = validate_history(history)
    date = pd.Timestamp(race_date)
    past = history.loc[history.race_date < date].copy()
    if past.empty:
        raise ValueError("No historical races before target date")
    if ((past.season == season) & (past["round"] == round_number)).any():
        raise ValueError("Target race key already exists in historical data")
    # Copy only allowed input columns: supplied targets/rolling features are never trusted.
    target = grid[["driver_id", "driver", "team_id", "team", "grid_position",
                   "qualifying_position", "q1", "q2", "q3"]].copy()
    target["race_date"], target["season"], target["round"] = date, season, round_number
    for col in ("finish_position", "points", "winner"):
        target[col] = np.nan
    combined = _engineer(pd.concat([past, target], ignore_index=True))
    return combined.loc[(combined.season == season) & (combined["round"] == round_number)].reset_index(drop=True)
