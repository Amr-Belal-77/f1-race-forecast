
from pathlib import Path
import json
import time

import pandas as pd
import requests

# =====================================
# 1. PROJECT CONFIGURATION
# =====================================

BASE_URL = "https://api.jolpi.ca/ergast/f1"

START_YEAR = 2018
END_YEAR = 2026

# Exclude the Sao Paulo race itself and future races
CUTOFF_DATE = "2026-11-08"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = RAW_DIR / "api_cache"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. DOWNLOAD ALL PAGES FROM THE API
# =====================================

def fetch_season(year, session_type):
    """
    Download all results for one season and session type.
    session_type = 'results' or 'qualifying'
    """

    records = []
    offset = 0
    page_size = 100

    if session_type == "results":
        results_key = "Results"
    else:
        results_key = "QualifyingResults"

    while True:
        cache_file = CACHE_DIR / (
            f"{year}_{session_type}_{offset}.json"
        )

        # Use saved API data if available
        if cache_file.exists():
            data = json.loads(
                cache_file.read_text(encoding="utf-8")
            )
        else:
            url = f"{BASE_URL}/{year}/{session_type}.json"

            for attempt in range(5):
                response = requests.get(
                    url,
                    params={
                        "limit": page_size,
                        "offset": offset
                    },
                    timeout=30
                )

                if response.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue

                response.raise_for_status()
                data = response.json()
                break
            else:
                raise RuntimeError("API rate limit exceeded")

            cache_file.write_text(
                json.dumps(data),
                encoding="utf-8"
            )

            # Respect the public API rate limit
            time.sleep(0.4)

        metadata = data["MRData"]
        total = int(metadata["total"])
        actual_limit = int(metadata["limit"])

        if actual_limit <= 0:
            raise ValueError("Invalid API page size")

        races = metadata["RaceTable"].get("Races", [])

        for race in races:
            race_date = race["date"]

            # No results from the target race or later
            if race_date >= CUTOFF_DATE:
                continue

            for result in race.get(results_key, []):
                driver = result["Driver"]
                team = result.get("Constructor", {})

                row = {
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_date": race_date,
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "driver_id": driver["driverId"],
                    "driver": (
                        driver["givenName"] + " "
                        + driver["familyName"]
                    ),
                    "team_id": team.get("constructorId"),
                    "team": team.get("name"),
                }

                if session_type == "results":
                    row.update({
                        "grid_position": result.get("grid"),
                        "finish_position": result.get("position"),
                        "points": result.get("points"),
                        "status": result.get("status")
                    })
                else:
                    row.update({
                        "qualifying_position": result.get(
                            "position"
                        ),
                        "q1": result.get("Q1"),
                        "q2": result.get("Q2"),
                        "q3": result.get("Q3")
                    })

                records.append(row)

        offset += actual_limit

        if offset >= total:
            break

    return records


# =====================================
# 3. COLLECT ALL HISTORICAL SEASONS
# =====================================

race_records = []
qualifying_records = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"\nCollecting season {year}...")

    races = fetch_season(year, "results")
    qualifying = fetch_season(year, "qualifying")

    race_records.extend(races)
    qualifying_records.extend(qualifying)

    print(f"Race result rows: {len(races)}")
    print(f"Qualifying rows: {len(qualifying)}")


# =====================================
# 4. CONVERT TO DATAFRAMES
# =====================================

race_df = pd.DataFrame(race_records)
qualifying_df = pd.DataFrame(qualifying_records)

if race_df.empty:
    raise ValueError("No historical race data downloaded")

# Qualifying columns we want to add
q_columns = [
    "season", "round", "driver_id",
    "qualifying_position", "q1", "q2", "q3"
]

qualifying_df = qualifying_df.reindex(columns=q_columns)


# =====================================
# 5. MERGE RACE AND QUALIFYING RESULTS
# =====================================

join_columns = ["season", "round", "driver_id"]

df = race_df.merge(
    qualifying_df,
    on=join_columns,
    how="left",
    validate="one_to_one"
)


# =====================================
# 6. CLEAN DATA TYPES
# =====================================

numeric_columns = [
    "grid_position",
    "qualifying_position",
    "finish_position",
    "points"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df["race_date"] = pd.to_datetime(df["race_date"])

# Target label
df["winner"] = (
    df["finish_position"] == 1
).astype(int)


# =====================================
# 7. VALIDATE THE DATA
# =====================================

assert not df.duplicated(join_columns).any(), (
    "Duplicate driver-race records found"
)

winners_per_race = df.groupby(
    ["season", "round"]
)["winner"].sum()

assert (winners_per_race == 1).all(), (
    "Some races do not have exactly one winner"
)

assert (df["race_date"] < pd.Timestamp(CUTOFF_DATE)).all(), (
    "Future race data detected"
)


# =====================================
# 8. SORT AND SAVE
# =====================================

df = df.sort_values(
    ["season", "round", "finish_position"]
).reset_index(drop=True)

output_file = RAW_DIR / "historical_f1_2018_2026.csv"

df.to_csv(output_file, index=False)


# =====================================
# 9. INSPECT OUR DATA
# =====================================

print("\n========== DATASET SUMMARY ==========")

print("Total driver-race records:", len(df))
print("Total historical races:", len(winners_per_race))
print("Number of drivers:", df["driver_id"].nunique())

print("\nRows per season:")
print(df.groupby("season").size())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst five records:")
print(df.head().to_string(index=False))

# First simple historical baseline
pole_starters = df[df["grid_position"] == 1]

if not pole_starters.empty:
    pole_win_rate = pole_starters["winner"].mean() * 100

    print(
        f"\nPole-position win rate: "
        f"{pole_win_rate:.2f}%"
    )

print("\nSaved dataset to:")
print(output_file)
