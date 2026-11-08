
from pathlib import Path

import requests
import pandas as pd

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Historical race: 2025 Sao Paulo Grand Prix
BASE_URL = "https://api.jolpi.ca/ergast/f1"
YEAR = 2025
ROUND = 21

# 1. Request race and qualifying results
def fetch_data(session):
    url = f"{BASE_URL}/{YEAR}/{ROUND}/{session}.json"
    response = requests.get(
        url, params={"limit": 100}, timeout=30
    )
    response.raise_for_status()
    return response.json()["MRData"]["RaceTable"]["Races"][0]

race = fetch_data("results")
qualifying = fetch_data("qualifying")

# 2. Build qualifying position lookup
qualifying_positions = {
    result["Driver"]["driverId"]: int(result["position"])
    for result in qualifying["QualifyingResults"]
}

# 3. Build one row per driver
records = []

for result in race["Results"]:
    driver = result["Driver"]
    driver_id = driver["driverId"]

    records.append({
        "year": YEAR,
        "round": ROUND,
        "race": race["raceName"],
        "driver_id": driver_id,
        "driver": (
            driver["givenName"] + " " + driver["familyName"]
        ),
        "team": result["Constructor"]["name"],
        "grid_position": int(result["grid"]),
        "qualifying_position": qualifying_positions.get(driver_id),
        "finish_position": int(result["position"]),
        "status": result["status"],
        "winner": int(result["position"] == "1")
    })

# 4. Convert to Pandas DataFrame
df = pd.DataFrame(records)

# 5. Validate the data
assert df["winner"].sum() == 1, "Expected one winner"
assert df["driver_id"].is_unique, "Duplicate driver found"
assert "São Paulo" in race["raceName"] or "Brazil" in race["raceName"]

# 6. Save as CSV
output = DATA_DIR / "sao_paulo_2025_results.csv"
df.to_csv(output, index=False)

# 7. Inspect the dataset
print(df.to_string(index=False))
print("\nDataset shape:", df.shape)
print("\nWinner:")
print(df.loc[df["winner"] == 1, "driver"].to_string(index=False))
print("\nSaved to:", output)
