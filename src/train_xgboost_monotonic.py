
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# =====================================
# 1. LOAD DATA
# =====================================

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT / "data" / "processed" / "f1_features.csv"
)

saved = joblib.load(
    ROOT / "models" / "f1_xgboost_model.joblib"
)

FEATURES = saved["features"]


# =====================================
# 2. HANDLE GRID POSITION ZERO
# =====================================

nonstandard = df["grid_position"] <= 0

df["pit_lane_start"] = nonstandard.astype(int)

max_grid = df.groupby(
    ["season", "round"]
)["grid_position"].transform("max")

df.loc[nonstandard, "grid_position"] = (
    max_grid[nonstandard] + 1
)


# =====================================
# 3. CHRONOLOGICAL SPLIT
# =====================================

train = df[df["season"] <= 2024].copy()
val = df[df["season"] == 2025].copy()
test = df[df["season"] == 2026].copy()


# =====================================
# 4. DEFINE MONOTONIC CONSTRAINTS
# =====================================

# -1 means: increasing this feature
# cannot increase the model's prediction.
#
# 0 means: no monotonic constraint.

constraints = [
    -1 if feature in [
        "grid_position",
        "qualifying_position"
    ] else 0
    for feature in FEATURES
]

# SimpleImputer adds missing-value indicators.
# These additional features have no constraints.

missing_indicators = int(
    train[FEATURES].isna().any().sum()
)

constraints += [0] * missing_indicators

print("Monotonic constraints:", constraints)


# =====================================
# 5. MODEL FACTORY
# =====================================

def create_model(monotonic=False):

    params = {
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "min_child_weight": 5,
        "reg_lambda": 10,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": 4
    }

    if monotonic:
        params["monotone_constraints"] = tuple(
            constraints
        )

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "classifier",
            XGBClassifier(**params)
        )
    ])


# =====================================
# 6. RACE-LEVEL EVALUATION
# =====================================

def evaluate(model, data, label):

    result = data[
        ["season", "round", "driver", "winner"]
    ].copy()

    raw = model.predict_proba(
        data[FEATURES]
    )[:, 1]

    result["raw_score"] = raw

    keys = ["season", "round"]

    totals = result.groupby(
        keys
    )["raw_score"].transform("sum")

    result["probability"] = (
        result["raw_score"] / totals
    )

    assert (
        result.groupby(keys)["winner"].sum() == 1
    ).all()

    predicted = result.loc[
        result.groupby(keys)["probability"].idxmax()
    ]

    accuracy = predicted["winner"].mean()

    actual_winners = result[
        result["winner"] == 1
    ]

    log_loss = -np.log(
        actual_winners["probability"].clip(
            lower=1e-12
        )
    ).mean()

    result["squared_error"] = (
        result["probability"] - result["winner"]
    ) ** 2

    brier = result.groupby(
        keys
    )["squared_error"].sum().mean()

    print(f"\n{label}")
    print(f"Top-1 Accuracy: {accuracy:.2%}")
    print(f"Race Log Loss: {log_loss:.4f}")
    print(f"Brier Score: {brier:.4f}")

    return accuracy, log_loss, brier


# =====================================
# 7. TRAIN BOTH MODELS
# =====================================

normal_model = create_model(monotonic=False)
constrained_model = create_model(monotonic=True)

print("\nTraining original XGBoost...")
normal_model.fit(
    train[FEATURES],
    train["winner"]
)

print("Training constrained XGBoost...")
constrained_model.fit(
    train[FEATURES],
    train["winner"]
)


# =====================================
# 8. COMPARE BOTH MODELS
# =====================================

for name, model in [
    ("Original XGBoost", normal_model),
    ("Constrained XGBoost", constrained_model)
]:

    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    evaluate(model, val, "2025 Validation")
    evaluate(model, test, "2026 Diagnostic")


# =====================================
# 9. TRAIN FINAL EXPERIMENTAL MODEL
# =====================================

# Includes all completed races in our dataset.
# The target Sao Paulo race is excluded.

constrained_model.fit(
    df[FEATURES],
    df["winner"]
)

output = (
    ROOT / "models"
    / "f1_xgboost_monotonic_model.joblib"
)

joblib.dump(
    {
        "model": constrained_model,
        "features": FEATURES
    },
    output
)

print("\nConstrained model saved to:", output)


# =====================================
# 10. VERIFY MONOTONIC BEHAVIOR
# =====================================

example = df[df["season"] == 2026].iloc[0]

scenarios = pd.DataFrame([
    example[FEATURES].to_dict()
    for _ in range(22)
])

scenarios["grid_position"] = np.arange(1, 23)

scores = constrained_model.predict_proba(
    scenarios[FEATURES]
)[:, 1]

increases = np.sum(
    np.diff(scores) > 1e-8
)

print("\nGrid sensitivity increases:", increases)

assert increases == 0, (
    "Grid monotonic constraint was violated"
)

print("Monotonic grid test passed!")
