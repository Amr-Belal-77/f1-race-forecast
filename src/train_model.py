
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ======================================
# 1. LOAD OUR FEATURE DATASET
# ======================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = ROOT / "data" / "processed" / "f1_features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

FEATURES = [
    "grid_position",
    "qualifying_position",
    "driver_avg_finish_last5",
    "driver_avg_points_last5",
    "driver_win_rate_last5",
    "driver_prior_starts",
    "team_avg_points_last5",
    "pit_lane_start"
]


# ======================================
# 2. HANDLE NONSTANDARD GRID POSITIONS
# ======================================

# Grid position 0 does not mean pole.
# Represent it as behind the numbered grid
# and create a flag identifying the situation.

invalid_grid = df["grid_position"] <= 0

df["pit_lane_start"] = invalid_grid.astype(int)

last_grid = df.groupby(
    ["season", "round"]
)["grid_position"].transform("max")

df.loc[invalid_grid, "grid_position"] = (
    last_grid[invalid_grid] + 1
)

print("Nonstandard grid positions:", invalid_grid.sum())


# ======================================
# 3. CHRONOLOGICAL DATA SPLIT
# ======================================

train = df[df["season"] <= 2024].copy()
validation = df[df["season"] == 2025].copy()
test = df[df["season"] == 2026].copy()

assert not train.empty
assert not validation.empty
assert not test.empty

print("\nTraining rows:", len(train))
print("Validation rows:", len(validation))
print("Test rows:", len(test))


# ======================================
# 4. CREATE OUR ML PIPELINE
# ======================================

model = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median",
            add_indicator=True
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])


# ======================================
# 5. TRAIN THE MODEL
# ======================================

X_train = train[FEATURES]
y_train = train["winner"]

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# ======================================
# 6. EVALUATE PER RACE
# ======================================

def evaluate_model(data, name):

    # Predict individual driver scores
    raw_probabilities = model.predict_proba(
        data[FEATURES]
    )[:, 1]

    results = data[
        ["season", "round", "driver",
         "grid_position", "winner"]
    ].copy()

    results["raw_probability"] = raw_probabilities

    race_keys = ["season", "round"]

    # Normalize probabilities within each race
    totals = results.groupby(
        race_keys
    )["raw_probability"].transform("sum")

    results["win_probability"] = (
        results["raw_probability"] / totals
    )

    # Every race must contain exactly one winner
    assert (
        results.groupby(race_keys)["winner"].sum() == 1
    ).all()

    # Select the most likely winner of each race
    predicted = results.loc[
        results.groupby(
            race_keys
        )["win_probability"].idxmax()
    ]

    # Baseline: always choose the pole sitter
    pole_predictions = results.loc[
        results.groupby(
            race_keys
        )["grid_position"].idxmin()
    ]

    accuracy = predicted["winner"].mean()
    pole_accuracy = pole_predictions["winner"].mean()

    # Probability assigned to actual winners
    actual_winners = results[
        results["winner"] == 1
    ]

    race_log_loss = -np.log(
        actual_winners["win_probability"].clip(
            lower=1e-12
        )
    ).mean()

    # Multiclass Brier score, averaged per race
    results["squared_error"] = (
        results["win_probability"] - results["winner"]
    ) ** 2

    brier = results.groupby(
        race_keys
    )["squared_error"].sum().mean()

    print(f"\n========== {name} ==========")
    print("Races evaluated:", len(predicted))
    print(f"Model Top-1 accuracy: {accuracy:.2%}")
    print(f"Pole baseline accuracy: {pole_accuracy:.2%}")
    print(f"Race Log Loss: {race_log_loss:.4f}")
    print(f"Multiclass Brier Score: {brier:.4f}")

    results.to_csv(
        OUTPUT_DIR / f"predictions_{name.lower()}.csv",
        index=False
    )


# Evaluate without retraining on the holdout data
evaluate_model(validation, "2025")
evaluate_model(test, "2026")


# ======================================
# 7. RETRAIN FOR TODAY'S PREDICTION
# ======================================

# After evaluation, use all completed races.
# The dataset excludes the target Sao Paulo race.

model.fit(df[FEATURES], df["winner"])

model_path = MODEL_DIR / "f1_logistic_model.joblib"

joblib.dump(
    {"model": model, "features": FEATURES},
    model_path
)

print("\nFinal model saved to:")
print(model_path)
