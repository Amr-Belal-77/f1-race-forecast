"""F1 Sao Paulo 2026: test whether grid-displacement features help XGBoost.

Run from VS Code project root:
    python src\train_grid_displacement_v3.py

This script does NOT predict the target race. It tests V2 vs V3 on earlier
seasons and saves an experimental V3 model for a later step.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "f1_features_v2.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RACE_KEYS = ["season", "round"]
DRIVER_RACE_KEYS = RACE_KEYS + ["driver_id"]
CUTOFF = pd.Timestamp("2026-11-08")

V2_FEATURES = [
    "grid_position",
    "qualifying_position",
    "driver_avg_finish_last5",
    "driver_avg_points_last5",
    "driver_win_rate_last5",
    "driver_prior_starts",
    "team_avg_points_last5",
    "pit_lane_start",
    "q1_gap_s",
    "q2_gap_s",
    "q3_gap_s",
    "has_q2_time",
    "has_q3_time",
]

# Derived from information known before lights-out; these are not official
# FIA penalty counts. Grid shifts can also be caused by other drivers' penalties.
NEW_FEATURES = [
    "grid_places_lost_vs_qualifying",
    "grid_places_gained_vs_qualifying",
    "starts_p14_or_worse",
]
V3_FEATURES = V2_FEATURES + NEW_FEATURES


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
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
        )),
    ])


def prepare_data(df):
    df = df.copy()
    df["race_date"] = pd.to_datetime(df["race_date"])
    if df.duplicated(DRIVER_RACE_KEYS).any():
        raise ValueError("Duplicate driver-race records in feature dataset")
    if not (df["race_date"] < CUTOFF).all():
        raise ValueError("Training data contains the Sao Paulo target race or later")
    wins_per_race = df.groupby(RACE_KEYS)["winner"].sum()
    if not (wins_per_race == 1).all():
        raise ValueError("Every completed historical race must have one winner")

    for col in ["grid_position", "qualifying_position"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Historical grid=0 is not pole; this follows our earlier V1/V2 approach.
    bad_grid = df["grid_position"].isna() | (df["grid_position"] <= 0)
    max_grid = df.groupby(RACE_KEYS)["grid_position"].transform("max")
    df["pit_lane_start"] = bad_grid.astype(int)
    df.loc[bad_grid, "grid_position"] = max_grid[bad_grid] + 1

    # A missing/invalid qualifying position is not an actual grid placement.
    valid_qual = df["qualifying_position"].where(df["qualifying_position"] > 0)
    change = df["grid_position"] - valid_qual
    df["grid_places_lost_vs_qualifying"] = change.clip(lower=0)
    df["grid_places_gained_vs_qualifying"] = (-change).clip(lower=0)
    df["starts_p14_or_worse"] = (df["grid_position"] >= 14).astype(int)

    missing = sorted(set(V3_FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns required for training: {missing}")
    return df


def predict_race_probabilities(model, fold, features):
    output = fold[
        ["season", "round", "driver_id", "driver", "grid_position",
         "qualifying_position", "grid_places_lost_vs_qualifying", "winner"]
    ].copy()
    output["score"] = model.predict_proba(fold[features])[:, 1]
    totals = output.groupby(RACE_KEYS)["score"].transform("sum")
    if not (totals > 0).all():
        raise ValueError("Cannot normalize zero-score race")
    output["probability"] = output["score"] / totals
    return output


def score_predictions(pred):
    chosen = pred.loc[pred.groupby(RACE_KEYS)["probability"].idxmax()]
    pole = pred.loc[pred.groupby(RACE_KEYS)["grid_position"].idxmin()]
    actual = pred.loc[pred["winner"] == 1]
    losses = -np.log(actual["probability"].clip(lower=1e-12))
    squared = (pred["probability"] - pred["winner"]) ** 2
    brier = squared.groupby([pred[k] for k in RACE_KEYS]).sum().mean()
    return {
        "races": len(chosen),
        "correct": int(chosen["winner"].sum()),
        "pole_correct": int(pole["winner"].sum()),
        "top1_accuracy": float(chosen["winner"].mean()),
        "log_loss": float(losses.mean()),
        "brier": float(brier),
    }


def print_group_audit(pred, title, mask):
    selected = pred.loc[mask]
    print(
        f"{title}: {len(selected)} driver-race rows; "
        f"expected wins={selected['probability'].sum():.2f}; "
        f"actual wins={int(selected['winner'].sum())}"
    )


def main():
    df = pd.read_csv(INPUT_FILE)
    df = prepare_data(df)
    print(f"Historical records: {len(df)}")
    print("New features:")
    print(df[NEW_FEATURES].describe().to_string())
    print("Missing new feature values:")
    print(df[NEW_FEATURES].isna().sum().to_string())

    all_metrics = []
    all_predictions = []
    for test_year in range(2022, 2027):
        train = df.loc[df["season"] < test_year]
        test = df.loc[df["season"] == test_year]
        if train.empty or test.empty:
            raise ValueError(f"Missing training or test data for {test_year}")
        for name, feats in [("V2", V2_FEATURES), ("V3_grid_displacement", V3_FEATURES)]:
            model = make_pipeline()
            model.fit(train[feats], train["winner"])
            pred = predict_race_probabilities(model, test, feats)
            metrics = score_predictions(pred)
            all_metrics.append({"year": test_year, "model": name, **metrics})
            pred["model"] = name
            all_predictions.append(pred)
            print(
                f"{test_year} {name:20s}: "
                f"{metrics['correct']}/{metrics['races']} winners; "
                f"log loss={metrics['log_loss']:.4f}; "
                f"Brier={metrics['brier']:.4f}"
            )

    metrics_df = pd.DataFrame(all_metrics)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df.to_csv(OUTPUT_DIR / "grid_displacement_v2_v3_metrics.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / "grid_displacement_v2_v3_predictions.csv", index=False)

    print("\n=== 2022-2025 SUMMARY (2026 DIAGNOSTIC EXCLUDED) ===")
    for model_name, g in metrics_df.loc[metrics_df["year"] < 2026].groupby("model"):
        total_races = int(g["races"].sum())
        print(
            f"{model_name:20s} | "
            f"{int(g['correct'].sum())}/{total_races} correct | "
            f"pole={int(g['pole_correct'].sum())}/{total_races} | "
            f"accuracy={g['correct'].sum() / total_races:.2%} | "
            f"log loss={np.average(g['log_loss'], weights=g['races']):.4f} | "
            f"Brier={np.average(g['brier'], weights=g['races']):.4f}"
        )

    print("\n=== HISTORICAL PENALTY/GRID-DISPLACEMENT AUDIT: 2022-2025 ===")
    for name in ["V2", "V3_grid_displacement"]:
        sub = pred_df.loc[(pred_df["season"] <= 2025) & (pred_df["model"] == name)]
        print(name)
        print_group_audit(sub, "P14+", sub["grid_position"] >= 14)
        print_group_audit(
            sub,
            "Lost >=5 places vs qualifying",
            sub["grid_places_lost_vs_qualifying"] >= 5,
        )

    # Save a candidate, not a declared winning model. Decide after reviewing metrics.
    candidate = make_pipeline()
    candidate.fit(df[V3_FEATURES], df["winner"])
    target = MODEL_DIR / "f1_xgboost_v3_grid_displacement.joblib"
    joblib.dump({"model": candidate, "features": V3_FEATURES,
                 "version": "experimental_grid_displacement"}, target)
    print("\nSaved experimental V3 model:", target)
    print("Saved evaluation CSVs in:", OUTPUT_DIR)
    print("NOTE: Grid displacement is not an official penalty count.")
    print("NOTE: Do not select a model based on 2026 diagnostics alone.")


if __name__ == "__main__":
    main()
