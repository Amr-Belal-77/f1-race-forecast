
from pathlib import Path
import unicodedata

import joblib
import numpy as np
import pandas as pd

# ======================================
# 1. PROJECT PATHS
# ======================================

ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    ROOT / "data" / "raw"
    / "historical_f1_2018_2026.csv"
)

MODEL_FILE = (
    ROOT / "models"
    / "f1_xgboost_model.joblib"
)

OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "sao_paulo_2026_xgboost_predictions.csv"
)

RACE_DATE = pd.Timestamp("2026-11-08")


# ======================================
# 2. STARTING GRID
# ======================================

# Format:
# (starting_position, surname, qualifying_position)
#
# This is our existing provisional grid.
# Verify against the latest FIA grid before
# using the predictions for race analysis.

GRID = [
    (1, "Norris", 1),
    (2, "Antonelli", 3),
    (3, "Lawson", 5),
    (4, "Hulkenberg", 6),
    (5, "Hadjar", 7),
    (6, "Bearman", 8),
    (7, "Russell", 2),
    (8, "Alonso", 9),
    (9, "Albon", 10),
    (10, "Piastri", 12),
    (11, "Camara", 13),
    (12, "Lindblad", 14),
    (13, "Sainz", 15),
    (14, "Leclerc", 4),
    (15, "Colapinto", 17),
    (16, "Gasly", 18),
    (17, "Stroll", 19),
    (18, "Bottas", 20),
    (19, "Bortoleto", 22),
    (20, "Hamilton", 11),
    (21, "Verstappen", 16),
    (22, "Perez", 21),
]


# ======================================
# 3. LOAD HISTORICAL DATA AND MODEL
# ======================================

print("Loading historical F1 data...")

history = pd.read_csv(RAW_FILE)

history["race_date"] = pd.to_datetime(
    history["race_date"]
)

# Exclude today's race and anything later
history = history[
    history["race_date"] < RACE_DATE
].copy()

history = history.sort_values(
    ["race_date", "season", "round"]
).reset_index(drop=True)

saved = joblib.load(MODEL_FILE)

model = saved["model"]
features = saved["features"]

print(f"Historical records loaded: {len(history)}")
print("Trained ML model loaded successfully!")


# ======================================
# 4. NORMALIZE DRIVER NAMES
# ======================================

def normalize_name(name):
    """
    Remove accents and standardize names.

    Example:
    Hülkenberg -> hulkenberg
    Pérez -> perez
    """

    name = unicodedata.normalize("NFKD", name)

    name = "".join(
        char for char in name
        if not unicodedata.combining(char)
    )

    return name.casefold().strip()


history["surname_key"] = history["driver"].apply(
    lambda name: normalize_name(name.split()[-1])
)


# ======================================
# 5. BUILD TODAY'S FEATURE ROWS
# ======================================

records = []

print("\nPreparing driver features...\n")

for grid_position, surname, quali_position in GRID:

    key = normalize_name(surname)

    driver_history = history[
        history["surname_key"] == key
    ].copy()

    # ----------------------------------
    # HANDLE DRIVERS WITH NO HISTORY
    # ----------------------------------

    if driver_history.empty:

        if key == "camara":

            print(
                "[NEW DRIVER] Rafael Camara - "
                "using Haas team performance."
            )

            driver_name = "Rafael Camara"
            team_id = "haas"
            team_name = "Haas F1 Team"

        else:

            raise ValueError(
                f"No historical data found for: {surname}"
            )

    else:

        # Make sure a surname matches only one driver
        if driver_history["driver_id"].nunique() != 1:

            raise ValueError(
                f"Ambiguous driver surname: {surname}"
            )

        latest = driver_history.iloc[-1]

        driver_name = latest["driver"]
        team_id = latest["team_id"]
        team_name = latest["team"]

    # ----------------------------------
    # DRIVER'S LAST 5 COMPLETED RACES
    # ----------------------------------

    last_five = driver_history.tail(5)

    driver_avg_finish = (
        last_five["finish_position"].mean()
    )

    driver_avg_points = (
        last_five["points"].mean()
    )

    driver_win_rate = (
        last_five["winner"].mean()
    )

    driver_prior_starts = len(driver_history)

    # ----------------------------------
    # TEAM'S LAST 5 COMPLETED RACES
    # ----------------------------------

    team_races = (
        history[history["team_id"] == team_id]
        .groupby(
            ["season", "round", "race_date"],
            as_index=False
        )["points"]
        .sum()
        .sort_values(
            ["race_date", "season", "round"]
        )
    )

    team_last_five = team_races.tail(5)

    team_avg_points = (
        team_last_five["points"].mean()
    )

    # ----------------------------------
    # CREATE ONE RECORD PER DRIVER
    # ----------------------------------

    records.append({

        # FIXED: use variables, not string literals
        "driver": driver_name,
        "team": team_name,

        "grid_position": grid_position,
        "qualifying_position": quali_position,

        "driver_avg_finish_last5": driver_avg_finish,
        "driver_avg_points_last5": driver_avg_points,
        "driver_win_rate_last5": driver_win_rate,

        "driver_prior_starts": driver_prior_starts,

        "team_avg_points_last5": team_avg_points,

        "pit_lane_start": 0
    })

    print(
        f"P{grid_position:02d} | "
        f"{driver_name:<22} | "
        f"{team_name}"
    )


# ======================================
# 6. CREATE AND VALIDATE DATAFRAME
# ======================================

prediction_df = pd.DataFrame(records)

print("\nValidating prediction dataset...")

# Check total drivers
assert len(prediction_df) == 22, (
    f"Expected 22 drivers, got {len(prediction_df)}"
)

# Check unique driver names
if not prediction_df["driver"].is_unique:

    duplicates = prediction_df[
        prediction_df["driver"].duplicated(keep=False)
    ]

    raise ValueError(
        "Duplicate driver names detected:\n"
        + duplicates.to_string(index=False)
    )

# Check grid positions
assert sorted(
    prediction_df["grid_position"].tolist()
) == list(range(1, 23)), (
    "Starting grid contains missing or duplicate positions"
)

# Check required ML features
missing_features = set(features) - set(
    prediction_df.columns
)

if missing_features:

    raise ValueError(
        f"Missing required features: {missing_features}"
    )

print("All 22 drivers validated successfully!")

print("\nMissing feature values:")
print(prediction_df[features].isna().sum())


# ======================================
# 7. GENERATE WIN PROBABILITIES
# ======================================

print("\nRunning ML predictions...")

X_today = prediction_df[features].copy()

# Obtain individual driver scores
raw_scores = model.predict_proba(
    X_today
)[:, 1]

prediction_df["raw_score"] = raw_scores

# Normalize across all drivers
total_score = raw_scores.sum()

if total_score <= 0:
    raise ValueError(
        "Invalid model scores: cannot normalize"
    )

prediction_df["win_probability"] = (
    raw_scores / total_score
)

prediction_df["win_probability_pct"] = (
    prediction_df["win_probability"] * 100
)

# Validate probabilities
assert np.isclose(
    prediction_df["win_probability"].sum(),
    1.0
), "Probabilities do not sum to 100%"

print("Predictions generated successfully!")


# ======================================
# 8. SORT BY WIN PROBABILITY
# ======================================

prediction_df = prediction_df.sort_values(
    by="win_probability",
    ascending=False
).reset_index(drop=True)

prediction_df.index = prediction_df.index + 1

prediction_df.index.name = "rank"


# ======================================
# 9. SAVE PREDICTIONS
# ======================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

prediction_df.to_csv(
    OUTPUT_FILE,
    index=True
)


# ======================================
# 10. DISPLAY FINAL RESULTS
# ======================================

print("\n" + "=" * 75)
print("2026 SAO PAULO GRAND PRIX")
print("MACHINE LEARNING WINNER PREDICTIONS")
print("=" * 75)

display_columns = [
    "driver",
    "team",
    "grid_position",
    "qualifying_position",
    "win_probability_pct"
]

print(
    prediction_df[
        display_columns
    ].to_string(
        index=True,
        formatters={
            "win_probability_pct": "{:.2f}%".format
        }
    )
)

print("\n" + "=" * 75)

predicted_winner = prediction_df.iloc[0]

print("PREDICTED RACE WINNER:")
print(predicted_winner["driver"])

print(
    "Estimated probability:",
    f"{predicted_winner['win_probability_pct']:.2f}%"
)

print(
    "\nTotal probability:",
    f"{prediction_df['win_probability_pct'].sum():.2f}%"
)

print("\nPrediction file saved to:")
print(OUTPUT_FILE)

print("\nNOTE:")
print(
    "Experimental Logistic Regression predictions. "
    "Model underperformed the pole baseline "
    "in historical holdout tests."
)
