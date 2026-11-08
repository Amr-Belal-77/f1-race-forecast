
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT / "data" / "processed" / "f1_features_v2.csv"
)

# 1. Prepare nonstandard grid positions
nonstandard = df["grid_position"] <= 0
df["pit_lane_start"] = nonstandard.astype(int)

max_grid = df.groupby(
    ["season", "round"]
)["grid_position"].transform("max")

df.loc[nonstandard, "grid_position"] = (
    max_grid[nonstandard] + 1
)

# 2. Feature groups
V1 = [
    "grid_position",
    "qualifying_position",
    "driver_avg_finish_last5",
    "driver_avg_points_last5",
    "driver_win_rate_last5",
    "driver_prior_starts",
    "team_avg_points_last5",
    "pit_lane_start",
]

V2 = V1 + [
    "q1_gap_s",
    "q2_gap_s",
    "q3_gap_s",
    "has_q2_time",
    "has_q3_time",
]

assert set(V2).issubset(df.columns)

# 3. Define model
def create_model():
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("classifier", XGBClassifier(
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
            n_jobs=4,
        ))
    ])


# 4. Evaluate an entire season
def evaluate(model, data, features):

    results = data[
        ["season", "round", "grid_position", "winner"]
    ].copy()

    results["score"] = model.predict_proba(
        data[features]
    )[:, 1]

    keys = ["season", "round"]

    total = results.groupby(keys)[
        "score"
    ].transform("sum")

    results["probability"] = results["score"] / total

    # Ensure one winner per race
    assert (
        results.groupby(keys)["winner"].sum() == 1
    ).all()

    # Model winner selections
    best = results.loc[
        results.groupby(keys)["probability"].idxmax()
    ]

    # Pole-sitter baseline
    pole = results.loc[
        results.groupby(keys)["grid_position"].idxmin()
    ]

    actual = results[results["winner"] == 1]

    log_loss = -np.log(
        actual["probability"].clip(lower=1e-12)
    ).mean()

    results["squared_error"] = (
        results["probability"] - results["winner"]
    ) ** 2

    brier = results.groupby(
        keys
    )["squared_error"].sum().mean()

    return {
        "races": len(best),
        "correct": int(best["winner"].sum()),
        "pole_correct": int(pole["winner"].sum()),
        "accuracy": best["winner"].mean(),
        "pole_accuracy": pole["winner"].mean(),
        "log_loss": log_loss,
        "brier": brier,
    }


# 5. Walk forward through seasons
records = []

for test_year in range(2022, 2027):

    train = df[df["season"] < test_year].copy()
    test = df[df["season"] == test_year].copy()

    for name, features in [
        ("XGBoost_V1", V1),
        ("XGBoost_V2", V2),
    ]:

        model = create_model()
        model.fit(train[features], train["winner"])

        metrics = evaluate(model, test, features)

        records.append({
            "season": test_year,
            "model": name,
            **metrics
        })

        print(
            f"{test_year} | {name} | "
            f"{metrics['correct']}/{metrics['races']} "
            f"correct | "
            f"Accuracy: {metrics['accuracy']:.2%} | "
            f"Log Loss: {metrics['log_loss']:.4f}"
        )


# 6. Save results
results_df = pd.DataFrame(records)

output = (
    ROOT / "data" / "processed"
    / "walk_forward_results.csv"
)

results_df.to_csv(output, index=False)

print("\n========== FULL RESULTS ==========")
print(results_df.to_string(index=False))


# 7. Aggregate earlier seasons separately
# We have already inspected 2026 repeatedly.
earlier = results_df[
    results_df["season"] < 2026
]

print("\n========== 2022-2025 SUMMARY ==========")

for name, group in earlier.groupby("model"):

    races = group["races"].sum()
    correct = group["correct"].sum()
    pole_correct = group["pole_correct"].sum()

    avg_loss = np.average(
        group["log_loss"],
        weights=group["races"]
    )

    avg_brier = np.average(
        group["brier"],
        weights=group["races"]
    )

    print(f"\n{name}")
    print(f"Races: {races}")
    print(f"Correct predictions: {correct}")
    print(f"Top-1 accuracy: {correct / races:.2%}")
    print(f"Pole accuracy: {pole_correct / races:.2%}")
    print(f"Race Log Loss: {avg_loss:.4f}")
    print(f"Brier Score: {avg_brier:.4f}")

print("\nSaved to:", output)
