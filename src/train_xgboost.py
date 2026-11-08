
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ======================================
# 1. LOAD DATA
# ======================================

ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    ROOT / "data" / "processed" / "f1_features.csv"
)

df = pd.read_csv(DATA_FILE)

# Reuse the exact Logistic Regression features
baseline = joblib.load(
    ROOT / "models" / "f1_logistic_model.joblib"
)

FEATURES = baseline["features"]


# ======================================
# 2. HANDLE NONSTANDARD GRID POSITIONS
# ======================================

nonstandard = df["grid_position"] <= 0

df["pit_lane_start"] = nonstandard.astype(int)

max_grid = df.groupby(
    ["season", "round"]
)["grid_position"].transform("max")

df.loc[nonstandard, "grid_position"] = (
    max_grid[nonstandard] + 1
)


# ======================================
# 3. CHRONOLOGICAL SPLIT
# ======================================

train = df[df["season"] <= 2024].copy()
val = df[df["season"] == 2025].copy()
test = df[df["season"] == 2026].copy()

print("Training:", len(train))
print("Validation:", len(val))
print("Test:", len(test))


# ======================================
# 4. XGBOOST MODEL
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
        "classifier",
        XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=10,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4
        )
    )
])


# ======================================
# 5. TRAIN
# ======================================

print("\nTraining XGBoost...")

model.fit(
    train[FEATURES],
    train["winner"]
)

print("Training completed!")


# ======================================
# 6. RACE-LEVEL EVALUATION
# ======================================

def evaluate(data, label):

    result = data[
        [
            "season", "round", "driver",
            "grid_position", "winner"
        ]
    ].copy()

    raw = model.predict_proba(
        data[FEATURES]
    )[:, 1]

    result["raw_score"] = raw

    keys = ["season", "round"]

    total = result.groupby(
        keys
    )["raw_score"].transform("sum")

    result["probability"] = (
        result["raw_score"] / total
    )

    # One winner per historical race
    assert (
        result.groupby(keys)["winner"].sum() == 1
    ).all()

    # Model's highest-probability driver
    predicted = result.loc[
        result.groupby(keys)["probability"].idxmax()
    ]

    # Pole-position baseline
    pole = result.loc[
        result.groupby(keys)["grid_position"].idxmin()
    ]

    model_accuracy = predicted["winner"].mean()
    pole_accuracy = pole["winner"].mean()

    winners = result[result["winner"] == 1]

    log_loss = -np.log(
        winners["probability"].clip(lower=1e-12)
    ).mean()

    result["squared_error"] = (
        result["probability"] - result["winner"]
    ) ** 2

    brier = result.groupby(
        keys
    )["squared_error"].sum().mean()

    print(f"\n========== {label} ==========")
    print("Races:", len(predicted))
    print(f"XGBoost Accuracy: {model_accuracy:.2%}")
    print(f"Pole Baseline: {pole_accuracy:.2%}")
    print(f"Race Log Loss: {log_loss:.4f}")
    print(f"Brier Score: {brier:.4f}")

    return result


# ======================================
# 7. EVALUATE
# ======================================

val_result = evaluate(val, "2025")
test_result = evaluate(test, "2026")


# ======================================
# 8. RETRAIN ON COMPLETED RACES
# ======================================

print("\nRetraining on all available races...")

model.fit(df[FEATURES], df["winner"])

MODEL_FILE = (
    ROOT / "models" / "f1_xgboost_model.joblib"
)

joblib.dump(
    {
        "model": model,
        "features": FEATURES
    },
    MODEL_FILE
)

print("\nSaved XGBoost model:")
print(MODEL_FILE)
