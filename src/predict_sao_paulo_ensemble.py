"""Combine existing pre-race 2026 Sao Paulo GP predictions from XGBoost V1 and V2.

Run from the project root:
    python src/predict_sao_paulo_ensemble.py

This script does not fit models or download data. It combines two PRE-RACE
probability distributions for the same 22-driver provisional starting grid.
It does NOT calibrate them. Verify any FIA grid updates before use.
"""

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
V1_FILE = PROCESSED / "sao_paulo_2026_xgboost_predictions.csv"
V2_FILE = PROCESSED / "sao_paulo_2026_v2_predictions.csv"
OUTPUT_FILE = PROCESSED / "sao_paulo_2026_ensemble_predictions.csv"
WEIGHT_V1 = 0.5
WEIGHT_V2 = 0.5


def clean_key(name: str) -> str:
    """Stable join key for accent differences in names."""
    expanded = unicodedata.normalize("NFKD", str(name))
    return "".join(ch for ch in expanded if not unicodedata.combining(ch)).casefold().strip()


def read_predictions(path: Path, version: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {version} predictions: {path}")
    frame = pd.read_csv(path)
    required = ["driver", "grid_position", "qualifying_position"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{version}: missing columns: {missing}")

    frame["driver_key"] = frame["driver"].map(clean_key)
    if len(frame) != 22 or frame["driver_key"].duplicated().any():
        raise ValueError(f"{version}: expected 22 unique drivers, got {len(frame)} rows")
    if sorted(frame["grid_position"].astype(int).tolist()) != list(range(1, 23)):
        raise ValueError(f"{version}: grid positions are not 1..22 exactly once")

    if "win_probability" in frame.columns:
        probabilities = pd.to_numeric(frame["win_probability"], errors="coerce")
    elif "win_probability_pct" in frame.columns:
        probabilities = pd.to_numeric(frame["win_probability_pct"], errors="coerce") / 100
    else:
        raise ValueError(
            f"{version}: expected win_probability or win_probability_pct column"
        )

    if not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError(f"{version}: non-finite probability detected")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{version}: probabilities outside [0, 1]")
    if not np.isclose(probabilities.sum(), 1.0, atol=0.002):
        raise ValueError(f"{version}: probabilities total {probabilities.sum():.6f}, expected 1")

    frame[f"{version}_probability"] = probabilities / probabilities.sum()
    keep = ["driver_key", "grid_position", "qualifying_position", f"{version}_probability"]
    if version == "v1":
        keep = ["driver", "team"] + keep
    return frame[keep].copy()


def main() -> None:
    if not np.isclose(WEIGHT_V1 + WEIGHT_V2, 1.0):
        raise ValueError("Ensemble weights must add up to 1")

    v1 = read_predictions(V1_FILE, "v1")
    v2 = read_predictions(V2_FILE, "v2")

    merged = v1.merge(
        v2,
        on="driver_key",
        how="outer",
        validate="one_to_one",
        suffixes=("_v1", "_v2"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Drivers differ between prediction files")
    for col in ("grid_position", "qualifying_position"):
        differing = merged[col + "_v1"] != merged[col + "_v2"]
        if differing.any():
            conflict = merged.loc[
                differing, ["driver", col + "_v1", col + "_v2"]
            ]
            raise ValueError(f"Grid/qualifying mismatch:\n{conflict.to_string(index=False)}")

    merged["grid_position"] = merged["grid_position_v1"].astype(int)
    merged["qualifying_position"] = merged["qualifying_position_v1"].astype(int)
    merged["ensemble_probability"] = (
        WEIGHT_V1 * merged["v1_probability"]
        + WEIGHT_V2 * merged["v2_probability"]
    )
    for label in ("v1", "v2", "ensemble"):
        merged[label + "_probability_pct"] = merged[label + "_probability"] * 100

    merged = merged.sort_values(
        "ensemble_probability", ascending=False
    ).reset_index(drop=True)
    merged.insert(0, "rank", np.arange(1, len(merged) + 1))
    columns = [
        "rank", "driver", "team", "grid_position", "qualifying_position",
        "v1_probability_pct", "v2_probability_pct", "ensemble_probability_pct",
    ]
    result = merged[columns].copy()
    result.to_csv(OUTPUT_FILE, index=False)

    print("\n2026 SAO PAULO GP — EXPERIMENTAL 50/50 ENSEMBLE")
    print("V1 + V2 pre-race predictions (same 22-driver grid)")
    print("=" * 86)
    print(result.to_string(
        index=False,
        formatters={
            col: "{:.2f}%".format for col in (
                "v1_probability_pct", "v2_probability_pct", "ensemble_probability_pct"
            )
        },
    ))
    print("\nSelected driver:", result.iloc[0]["driver"])
    print("Ensemble score: {:.2f}%".format(result.iloc[0]["ensemble_probability_pct"]))
    print("Total: {:.4f}%".format(result["ensemble_probability_pct"].sum()))
    print("Saved:", OUTPUT_FILE)
    print("NOTE: Uncalibrated experimental probabilities. The pole-sitter baseline remains competitive.")
    print("Verify any later FIA grid amendments before using this snapshot.")


if __name__ == "__main__":
    main()
