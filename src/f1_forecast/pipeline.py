"""Configurable artifact pipeline with input hashes and training provenance."""

import hashlib
import json
import platform
import tomllib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import validate_history
from .features import FEATURE_SETS, build_features, build_target_features
from .modeling import backtest, calibration_by_grid, fit_models, predict_models

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config/sao_paulo_2026.toml"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    for section in ("data", "race", "evaluation"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")
    data, race, evaluation = config["data"], config["race"], config["evaluation"]
    for section, fields in [
        (data, ["history", "grid", "cutoff"]),
        (race, ["name", "date", "season", "round"]),
        (evaluation, ["years", "diagnostic_from"]),
    ]:
        missing = set(fields) - set(section)
        if missing:
            raise ValueError(f"Missing config keys: {sorted(missing)}")
    if pd.Timestamp(data["cutoff"]) > pd.Timestamp(race["date"]):
        raise ValueError("Training cutoff must be on or before the target race date")
    if pd.Timestamp(race["date"]).year != race["season"]:
        raise ValueError("Target date and season disagree")
    if not isinstance(race["round"], int) or race["round"] < 1:
        raise ValueError("Race round must be a positive integer")
    years = evaluation["years"]
    if not years or any(not isinstance(y, int) or y < 1950 for y in years):
        raise ValueError("Evaluation years must contain valid seasons")
    for name in ("history", "grid"):
        data[name] = (path.parent / data[name]).resolve()
    config["config_path"] = path
    return config


def prepare_features(config, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    history = validate_history(pd.read_csv(config["data"]["history"]), config["data"]["cutoff"])
    features = build_features(history)
    features.to_csv(output / "features.csv", index=False)
    return history, features


def train(config, output, features, n_estimators=250):
    models = fit_models(features, n_estimators)
    directory = Path(output) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(
            {
                "model": model,
                "features": FEATURE_SETS[name],
                "cutoff": config["data"]["cutoff"],
                "history_sha256": file_hash(config["data"]["history"]),
                "training_max_date": features.race_date.max().strftime("%Y-%m-%d"),
                "n_estimators": n_estimators,
            },
            directory / f"{name}.joblib",
        )
    return models


def evaluate(config, output, features, n_estimators=250):
    summary, predictions = backtest(features, config["evaluation"]["years"], n_estimators)
    boundary = config["evaluation"]["diagnostic_from"]
    summary["sample"] = np.where(summary.season < boundary, "historical", "diagnostic")
    summary.to_csv(Path(output) / "backtest_summary.csv", index=False)
    predictions.to_csv(Path(output) / "backtest_predictions.csv", index=False)
    calibration_by_grid(predictions, boundary).to_csv(
        Path(output) / "grid_calibration.csv", index=False
    )
    return summary


def predict(config, output, history, models=None):
    output = Path(output)
    if models is None:
        models = {}
        for name in FEATURE_SETS:
            # Only load locally trained/trusted model artifacts, never uploaded files.
            bundle = joblib.load(output / "models" / f"{name}.joblib")
            if (
                bundle["history_sha256"] != file_hash(config["data"]["history"])
                or bundle["cutoff"] != config["data"]["cutoff"]
                or bundle["features"] != FEATURE_SETS[name]
            ):
                raise ValueError("Model provenance differs from config/history; rerun train")
            models[name] = bundle["model"]
    race = config["race"]
    target = build_target_features(
        history, pd.read_csv(config["data"]["grid"]), race["date"], race["season"], race["round"]
    )
    target.to_csv(output / "target_features.csv", index=False)
    result = predict_models(models, target).sort_values("Blend_50_50", ascending=False)
    result.insert(0, "rank", range(1, len(result) + 1))
    result.to_csv(output / "predictions.csv", index=False)
    return result


def write_manifest(config, output, history, n_estimators):
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "race": config["race"],
        "cutoff_exclusive": config["data"]["cutoff"],
        "training_rows": len(history),
        "training_races": len(history.groupby(["season", "round"])),
        "training_max_date": history.race_date.max().strftime("%Y-%m-%d"),
        "evaluation": config["evaluation"],
        "provenance": config["data"].get(
            "provenance", "User-supplied inputs; independently unverified"
        ),
        "score_status": "Uncalibrated normalized model scores; V3 is experimental",
        "inputs": {
            name: {"name": config["data"][name].name, "sha256": file_hash(config["data"][name])}
            for name in ("history", "grid")
        },
        "config_sha256": file_hash(config["config_path"]),
        "versions": {
            name: version(name) for name in ["numpy", "pandas", "scikit-learn", "xgboost"]
        },
        "python": platform.python_version(),
        "models": {
            name: {"features": columns, "n_estimators": n_estimators, "random_state": 42}
            for name, columns in FEATURE_SETS.items()
        },
        "artifacts": {
            name: file_hash(Path(output) / name)
            for name in ["predictions.csv", "backtest_summary.csv", "grid_calibration.csv"]
        },
    }
    Path(output, "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_pipeline(config, output="artifacts", n_estimators=250):
    output = Path(output).resolve()
    history, features = prepare_features(config, output)
    print(f"Validated {len(history)} historical driver-race rows", flush=True)
    evaluate(config, output, features, n_estimators)
    print("Walk-forward evaluation saved", flush=True)
    models = train(config, output, features, n_estimators)
    predictions = predict(config, output, history, models)
    write_manifest(config, output, history, n_estimators)
    print(f"Top ensemble selection: {predictions.iloc[0].driver} (uncalibrated score)", flush=True)
    print(f"Artifacts: {output}", flush=True)
    return output
