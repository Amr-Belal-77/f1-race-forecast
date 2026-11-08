
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ======================================
# 1. LOAD MODEL AND PREDICTIONS
# ======================================

ROOT = Path(__file__).resolve().parent.parent

saved = joblib.load(
    ROOT / "models" / "f1_xgboost_model.joblib"
)

model = saved["model"]
features = saved["features"]

df = pd.read_csv(
    ROOT / "data" / "processed"
    / "sao_paulo_2026_xgboost_predictions.csv"
)

DRIVERS = [
    "Max Verstappen",
    "George Russell",
    "Andrea Kimi Antonelli",
    "Lando Norris"
]

# ======================================
# 2. TEST DIFFERENT GRID POSITIONS
# ======================================

for driver in DRIVERS:

    original = df[df["driver"] == driver].copy()

    if len(original) != 1:
        raise ValueError(
            f"Expected exactly one row for {driver}"
        )

    actual_grid = int(
        original.iloc[0]["grid_position"]
    )

    # Create 22 hypothetical starting positions
    scenarios = pd.concat(
        [
            original.assign(grid_position=position)
            for position in range(1, 23)
        ],
        ignore_index=True
    )

    # Keep all other features unchanged
    raw_scores = model.predict_proba(
        scenarios[features]
    )[:, 1]

    print(f"\n{'=' * 55}")
    print(f"DRIVER: {driver}")
    print(f"Actual starting position: P{actual_grid}")
    print("Hypothetical grid -> raw model score")
    print(f"{'=' * 55}")

    for grid in [1, 2, 3, 5, 7, 10, 14, 18, 21, 22]:
        score = raw_scores[grid - 1]

        print(
            f"Grid P{grid:02d} -> "
            f"{score * 100:.4f}%"
        )

    # Test whether moving backward can
    # unexpectedly increase the model score
    increases = np.sum(
        np.diff(raw_scores) > 1e-9
    )

    print(
        "\nNumber of score increases "
        "when moving backward:",
        int(increases)
    )

    if increases > 0:
        print(
            "WARNING: Non-monotonic grid behavior detected"
        )
    else:
        print(
            "Grid response is non-increasing "
            "for these scenarios"
        )

print("\nSensitivity analysis completed.")
