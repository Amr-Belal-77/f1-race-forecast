"""Fixed research models and season-wise evaluation; no tuning on target outcomes."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .data import RACE_KEYS
from .features import FEATURE_SETS


def make_model(n_estimators=250):
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=n_estimators,
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
                ),
            ),
        ]
    )


def normalize_scores(scores, frame):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != len(frame) or not len(values):
        raise ValueError("Expected one score per driver in a nonempty race dataset")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Model scores must be finite and within [0, 1]")
    series = pd.Series(values, index=frame.index)
    totals = series.groupby([frame[k] for k in RACE_KEYS]).transform("sum")
    if totals.isna().any() or not totals.gt(0).all():
        raise ValueError("Cannot normalize a race with zero or missing total score")
    return series / totals


def score_races(frame, column):
    chosen = frame.loc[frame.groupby(RACE_KEYS)[column].idxmax()]
    actual = frame.loc[frame.winner.eq(1), column].clip(lower=1e-12)
    squared = (frame[column] - frame.winner) ** 2
    return {
        "races": len(chosen),
        "correct": int(chosen.winner.sum()),
        "top1_accuracy": float(chosen.winner.mean()),
        "race_log_loss": float(-np.log(actual).mean()),
        "brier": float(squared.groupby([frame[k] for k in RACE_KEYS]).sum().mean()),
    }


def fit_models(features, n_estimators=250):
    if features.winner.nunique() != 2:
        raise ValueError("Training requires both winner and non-winner examples")
    models = {}
    for name, columns in FEATURE_SETS.items():
        model = make_model(n_estimators)
        model.fit(features[columns], features.winner)
        models[name] = model
    return models


def predict_models(models, features):
    columns = [
        *RACE_KEYS,
        "driver_id",
        "driver",
        "team",
        "grid_position",
        "qualifying_position",
        "pit_lane_start",
        "grid_places_lost_vs_qualifying",
    ]
    if "winner" in features and features.winner.notna().all():
        columns.append("winner")
    result = features[columns].copy()
    for name, model in models.items():
        result[name] = normalize_scores(
            model.predict_proba(features[FEATURE_SETS[name]])[:, 1], features
        )
    result["Blend_50_50"] = 0.5 * result.V1 + 0.5 * result.V2
    return result


def backtest(features, years, n_estimators=250):
    if not years:
        raise ValueError("At least one evaluation fold is required")
    rows, outputs = [], []
    for year in sorted(set(years)):
        train = features.loc[features.season < year]
        test = features.loc[features.season == year].reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError(f"Empty training or evaluation fold for {year}")
        if train.race_date.max() >= test.race_date.min():
            raise ValueError(f"Non-chronological evaluation fold for {year}")
        models = fit_models(train, n_estimators)
        prediction = predict_models(models, test)
        metadata = {
            "season": year,
            "training_last_season": int(train.season.max()),
            "training_max_date": train.race_date.max().strftime("%Y-%m-%d"),
        }
        for name in [*FEATURE_SETS, "Blend_50_50"]:
            rows.append({**metadata, "model": name, **score_races(prediction, name)})
        pole = prediction.loc[prediction.groupby(RACE_KEYS).grid_position.idxmin()]
        rows.append(
            {
                **metadata,
                "model": "Pole_Top1",
                "races": len(pole),
                "correct": int(pole.winner.sum()),
                "top1_accuracy": float(pole.winner.mean()),
                "race_log_loss": np.nan,
                "brier": np.nan,
            }
        )
        outputs.append(prediction)
    return pd.DataFrame(rows), pd.concat(outputs, ignore_index=True)


def calibration_by_grid(predictions, diagnostic_from):
    frame = predictions.copy()
    frame["sample"] = np.where(frame.season < diagnostic_from, "historical", "diagnostic")
    frame["grid_band"] = pd.cut(
        frame.grid_position,
        [0, 1, 3, 6, 10, 13, np.inf],
        labels=["P1", "P2–P3", "P4–P6", "P7–P10", "P11–P13", "P14+"],
    )
    records = []
    for (sample, band), group in frame.groupby(["sample", "grid_band"], observed=True):
        for model in [*FEATURE_SETS, "Blend_50_50"]:
            records.append(
                {
                    "sample": sample,
                    "grid_band": band,
                    "model": model,
                    "driver_rows": len(group),
                    "actual_wins": int(group.winner.sum()),
                    "predicted_wins": float(group[model].sum()),
                    "mean_score": float(group[model].mean()),
                    "observed_win_rate": float(group.winner.mean()),
                }
            )
    return pd.DataFrame(records)
