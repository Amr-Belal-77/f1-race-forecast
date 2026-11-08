"""Load consistent, portable dashboard reports without deserializing model files."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import file_hash


def load_reports(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    filenames = ["predictions.csv", "backtest_summary.csv", "grid_calibration.csv"]
    for filename in filenames:
        expected = manifest.get("artifacts", {}).get(filename)
        if not expected or file_hash(directory / filename) != expected:
            raise ValueError(f"Report mismatch for {filename}; rerun the full pipeline")
    frames = [pd.read_csv(directory / filename) for filename in filenames]
    return manifest, *frames


def blend_predictions(predictions, v1_weight):
    if not np.isfinite(v1_weight) or not 0 <= v1_weight <= 1:
        raise ValueError("Ensemble weight must be within [0, 1]")
    frame = predictions.copy()
    frame["ensemble_score"] = v1_weight * frame.V1 + (1 - v1_weight) * frame.V2
    return frame.sort_values("ensemble_score", ascending=False).reset_index(drop=True)


def aggregate_summary(summary):
    records = []
    for model, group in summary.groupby("model", sort=False):
        races, correct = int(group.races.sum()), int(group.correct.sum())
        records.append(
            {
                "Model": model,
                "Races": races,
                "Correct": correct,
                "Winner accuracy (%)": 100 * correct / races,
                "Race log loss": (
                    np.average(group.race_log_loss, weights=group.races)
                    if model != "Pole_Top1"
                    else np.nan
                ),
                "Brier score": (
                    np.average(group.brier, weights=group.races) if model != "Pole_Top1" else np.nan
                ),
            }
        )
    return pd.DataFrame(records).set_index("Model")
