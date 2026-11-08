"""Input contracts and optional cached Jolpica data collection."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

RACE_KEYS = ["season", "round"]
ROW_KEYS = RACE_KEYS + ["driver_id"]
IDENTITY_COLUMNS = ["driver_id", "driver", "team_id", "team"]
BASE_URL = "https://api.jolpi.ca/ergast/f1"


def require_columns(frame, columns):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")


def validate_identity(frame):
    require_columns(frame, IDENTITY_COLUMNS)
    for col in IDENTITY_COLUMNS:
        if frame[col].isna().any() or frame[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identity: {col}")


def validate_history(frame, cutoff=None):
    """Return typed history, rejecting partial races, duplicate rows and future outcomes."""
    if frame.empty:
        raise ValueError("Historical dataset is empty")
    require_columns(
        frame,
        ROW_KEYS
        + [
            "race_date",
            "grid_position",
            "qualifying_position",
            "finish_position",
            "points",
            "winner",
            "q1",
            "q2",
            "q3",
        ],
    )
    validate_identity(frame)
    frame = frame.copy()
    frame["race_date"] = pd.to_datetime(frame.race_date, errors="raise", format="ISO8601")
    if frame.race_date.isna().any():
        raise ValueError("Missing race_date")
    if cutoff is not None and not frame.race_date.lt(pd.Timestamp(cutoff)).all():
        raise ValueError("Historical outcomes must be strictly before the cutoff")
    for col in [
        "season",
        "round",
        "finish_position",
        "winner",
        "grid_position",
        "qualifying_position",
        "points",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        values = frame[col].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {col}")
    for col in ["season", "round", "finish_position"]:
        if frame[col].isna().any() or not ((frame[col] > 0) & (frame[col] % 1 == 0)).all():
            raise ValueError(f"Invalid {col}")
        frame[col] = frame[col].astype(int)
    for col in ["grid_position", "qualifying_position"]:
        values = frame[col].dropna()
        if not (values.ge(0) & (values % 1 == 0)).all():
            raise ValueError(f"Invalid {col}: expected nonnegative integer positions or missing")
    if frame.points.lt(0).any():
        raise ValueError("Invalid points: race points cannot be negative")
    if frame.points.isna().any() or not frame.winner.isin([0, 1]).all():
        raise ValueError("Invalid points or winner labels")
    if not frame.winner.eq(frame.finish_position.eq(1).astype(int)).all():
        raise ValueError("Winner labels disagree with finish positions")
    if frame.duplicated(ROW_KEYS).any():
        raise ValueError("Duplicate driver-race records")
    grouped = frame.groupby(RACE_KEYS)
    if not grouped.winner.sum().eq(1).all():
        raise ValueError("Each historical race must have exactly one winner")
    if not grouped.race_date.nunique().eq(1).all():
        raise ValueError("Conflicting dates within a race")
    if not frame.race_date.dt.year.eq(frame.season).all():
        raise ValueError("Historical season disagrees with race date year")
    calendar = frame[RACE_KEYS + ["race_date"]].drop_duplicates()
    if calendar.race_date.duplicated().any():
        raise ValueError("Distinct races sharing a date are unsupported by date-only chronology")
    return frame.sort_values(["race_date", *ROW_KEYS]).reset_index(drop=True)


def _fetch_season(session, year, kind, cutoff, cache_dir, refresh):
    records, offset = [], 0
    while True:
        cache = cache_dir / f"{year}_{kind}_{offset}.json"
        if cache.is_file() and not refresh:
            data = json.loads(cache.read_text(encoding="utf-8"))
        else:
            response = session.get(
                f"{BASE_URL}/{year}/{kind}.json",
                params={"limit": 100, "offset": offset},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            cache.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.5)
        metadata = data["MRData"]
        limit, total = int(metadata["limit"]), int(metadata["total"])
        if limit <= 0:
            raise ValueError("Invalid API page size")
        key = "Results" if kind == "results" else "QualifyingResults"
        for race in metadata["RaceTable"].get("Races", []):
            if pd.Timestamp(race["date"]) >= pd.Timestamp(cutoff):
                continue
            for result in race.get(key, []):
                driver = result["Driver"]
                row = {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "driver_id": driver["driverId"],
                }
                if kind == "results":
                    team = result["Constructor"]
                    row.update(
                        {
                            "race_date": race["date"],
                            "race_name": race["raceName"],
                            "circuit_id": race["Circuit"]["circuitId"],
                            "driver": driver["givenName"] + " " + driver["familyName"],
                            "team_id": team["constructorId"],
                            "team": team["name"],
                            "grid_position": result.get("grid"),
                            "finish_position": result["position"],
                            "points": result["points"],
                            "status": result.get("status"),
                        }
                    )
                else:
                    row.update(
                        {
                            "qualifying_position": result.get("position"),
                            "q1": result.get("Q1"),
                            "q2": result.get("Q2"),
                            "q3": result.get("Q3"),
                        }
                    )
                records.append(row)
        offset += limit
        if offset >= total:
            return records


def collect_history(start, end, cutoff, cache_dir, refresh=False):
    """Collect completed races only; refresh explicitly to replace cached API responses."""
    if start > end:
        raise ValueError("Start season must not exceed end season")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results, qualifying = [], []
    retries = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retries))
        for year in range(start, end + 1):
            results.extend(_fetch_season(session, year, "results", cutoff, cache_dir, refresh))
            qualifying.extend(
                _fetch_season(session, year, "qualifying", cutoff, cache_dir, refresh)
            )
    if not results:
        raise ValueError("No historical race results returned")
    qcols = ROW_KEYS + ["qualifying_position", "q1", "q2", "q3"]
    frame = pd.DataFrame(results).merge(
        pd.DataFrame(qualifying).reindex(columns=qcols),
        on=ROW_KEYS,
        how="left",
        validate="one_to_one",
    )
    frame["winner"] = pd.to_numeric(frame.finish_position).eq(1).astype(int)
    return validate_history(frame, cutoff)
