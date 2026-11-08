"""Step 21: Audit historical grid risk versus our current São Paulo ensemble.

Run from the project root:
    python src/audit_grid_risk.py

This script is descriptive. It does NOT modify or recalibrate model scores.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HISTORY_FILE = ROOT / "data" / "raw" / "historical_f1_2018_2026.csv"
PREDICTIONS_FILE = (
    ROOT / "data" / "processed" / "sao_paulo_2026_ensemble_predictions.csv"
)
CUTOFF = pd.Timestamp("2026-11-08")

history = pd.read_csv(HISTORY_FILE)
history["race_date"] = pd.to_datetime(history["race_date"])
history = history.loc[history["race_date"] < CUTOFF].copy()
history["grid_position"] = pd.to_numeric(
    history["grid_position"], errors="coerce"
)
history["winner"] = pd.to_numeric(history["winner"], errors="raise")

race_keys = ["season", "round"]
unique_races = history[race_keys].drop_duplicates()
num_races = len(unique_races)
assert num_races > 0, "No historical races found."
assert (history.groupby(race_keys)["winner"].sum() == 1).all(), (
    "Expected exactly one winner per completed race."
)

pred = pd.read_csv(PREDICTIONS_FILE)
assert len(pred) == 22, f"Expected 22 drivers, found {len(pred)}"
assert pred["driver"].is_unique, "Driver names must be unique"
assert sorted(pred["grid_position"].astype(int)) == list(range(1, 23)), (
    "Starting positions must be 1 to 22"
)

prob = pred["ensemble_probability_pct"].astype(str).str.strip()
prob = pd.to_numeric(prob.str.replace("%", "", regex=False), errors="raise")
# Handles either percentages (e.g. 35.46) or fractions (e.g. 0.3546).
if np.isclose(prob.sum(), 1.0, atol=0.01):
    prob *= 100
assert np.isclose(prob.sum(), 100.0, atol=0.1), (
    f"Ensemble probabilities sum to {prob.sum():.4f}, not 100%"
)
pred["ensemble_pct"] = prob

# Only recorded positive starting-grid positions are interpreted as a grid place.
# A zero in the historical API can indicate a pit-lane or nonstandard start.
valid = history.loc[history["grid_position"].between(1, 22)].copy()

BINS = [0, 1, 2, 3, 6, 10, 13, 22]
LABELS = ["P1", "P2", "P3", "P4-P6", "P7-P10", "P11-P13", "P14-P22"]
valid["grid_band"] = pd.cut(valid["grid_position"], bins=BINS, labels=LABELS)
pred["grid_band"] = pd.cut(
    pd.to_numeric(pred["grid_position"]), bins=BINS, labels=LABELS
)

hist_by_band = valid.groupby("grid_band", observed=False).agg(
    historical_starts=("winner", "size"),
    historical_wins=("winner", "sum"),
)
hist_by_band["historic_win_share_pct"] = (
    hist_by_band["historical_wins"] / num_races * 100
)
hist_by_band["win_rate_per_start_pct"] = (
    hist_by_band["historical_wins"] / hist_by_band["historical_starts"] * 100
)

current_mass = pred.groupby("grid_band", observed=False)["ensemble_pct"].sum()
hist_by_band["today_ensemble_mass_pct"] = current_mass
hist_by_band = hist_by_band.reset_index()

print("\nF1 GRID POSITION RISK AUDIT")
print("=" * 75)
print(f"Historical races: {num_races}")
print(f"Historical records with nonstandard/invalid grid: {len(history) - len(valid)}")
print("\nGrid-band comparison:")
print(hist_by_band.to_string(index=False, formatters={
    "historic_win_share_pct": "{:.2f}%".format,
    "win_rate_per_start_pct": "{:.2f}%".format,
    "today_ensemble_mass_pct": "{:.2f}%".format,
}))

p21 = valid.loc[valid["grid_position"] == 21]
print(f"\nHistorical P21 winners: {int(p21['winner'].sum())} / {len(p21)} starts")
back = valid.loc[valid["grid_position"] >= 14]
print(f"Historical winners from P14+: {int(back['winner'].sum())} / {num_races} races")
print("\nToday's drivers with the highest forecast probability from P14 or worse:")
print(pred.loc[pred["grid_position"] >= 14, [
    "driver", "grid_position", "ensemble_pct"
]].sort_values("ensemble_pct", ascending=False).head(5).to_string(
    index=False, formatters={"ensemble_pct": "{:.2f}%".format}
))

output = ROOT / "data" / "processed" / "grid_risk_audit.csv"
output.parent.mkdir(parents=True, exist_ok=True)
hist_by_band.to_csv(output, index=False)
print(f"\nSaved summary: {output}")
print("\nNOTE: Historical frequencies are descriptive, not calibrated today's odds.")
print("Differences may reflect driver/car strength, penalties, and changing conditions.")
