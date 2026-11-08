"""Predict the 2026 Sao Paulo GP winner using the trained XGBoost V2 model.

Uses the pre-race driver/team features from the earlier prediction CSV,
plus Q1/Q2/Q3 times from FIA Final Qualifying Classification, Doc 50
(7 Nov 2026) and grid positions from FIA Provisional Starting Grid,
Doc 51 (7 Nov 2026). This is a frozen pre-race snapshot.

Qualifying: https://www.fia.com/system/files/decision-document/2026_sao_paulo_grand_prix_-_final_qualifying_classification.pdf
Starting grid: https://www.fia.com/system/files/decision-document/2026_sao_paulo_grand_prix_-_provisional_starting_grid.pdf
"""

from pathlib import Path
import unicodedata

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "sao_paulo_2026_xgboost_predictions.csv"
MODEL_FILE = ROOT / "models" / "f1_xgboost_v2_model.joblib"
OUTPUT_FILE = ROOT / "data" / "processed" / "sao_paulo_2026_v2_predictions.csv"

# Position on the provisional GRID, surname, qualifying POSITION, Q1, Q2, Q3.
# None = no valid recorded lap time in that qualifying segment.
# Snapshot: FIA Docs 50 and 51, published November 7, 2026.
QUALIFYING = [
    ( 1, "Norris",      1, "1:30.939", "1:29.207", "1:27.470"),
    ( 2, "Antonelli",   3, "1:32.229", "1:29.302", "1:28.095"),
    ( 3, "Lawson",      5, "1:32.085", "1:27.049", "1:28.321"),
    ( 4, "Hulkenberg",  6, "1:31.693", "1:28.361", "1:28.668"),
    ( 5, "Hadjar",      7, "1:32.059", "1:27.218", "1:28.763"),
    ( 6, "Bearman",     8, "1:33.030", "1:29.130", "1:29.831"),
    ( 7, "Russell",     2, "1:31.206", "1:26.745", "1:27.674"),
    ( 8, "Alonso",      9, "1:32.092", "1:29.072", "1:30.311"),
    ( 9, "Albon",      10, "1:33.150", "1:28.918", "1:30.456"),
    (10, "Piastri",    12, "1:32.251", "1:30.369", None),
    (11, "Camara",     13, "1:32.809", "1:31.515", None),
    (12, "Lindblad",   14, "1:32.306", "1:31.687", None),
    (13, "Sainz",      15, "1:32.959", "1:31.785", None),
    (14, "Leclerc",     4, "1:32.121", "1:28.082", "1:28.289"),
    (15, "Colapinto",  17, "1:33.327", None,       None),
    (16, "Gasly",      18, "1:33.449", None,       None),
    (17, "Stroll",     19, "1:33.653", None,       None),
    (18, "Bottas",     20, "1:33.774", None,       None),
    (19, "Bortoleto",  22, "1:41.564", None,       None),
    (20, "Hamilton",   11, "1:31.666", "1:30.001", None),
    (21, "Verstappen", 16, "1:31.238", None,       None),
    (22, "Perez",      21, "1:34.509", None,       None),
]


def surname_key(full_name):
    surname = str(full_name).strip().split()[-1]
    ascii_name = unicodedata.normalize("NFKD", surname)
    return "".join(
        c for c in ascii_name if not unicodedata.combining(c)
    ).casefold()


def lap_to_seconds(value):
    if value is None or pd.isna(value):
        return np.nan
    minutes, seconds = str(value).split(":")
    return int(minutes) * 60 + float(seconds)


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. Run the original XGBoost prediction script first."
        )
    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_FILE}. Run train_xgboost_v2.py first."
        )

    # The earlier CSV contains pre-race features (not race outcomes).
    base = pd.read_csv(INPUT_FILE)
    base["driver_key"] = base["driver"].map(surname_key)

    if len(base) != 22 or base["driver_key"].duplicated().any():
        raise ValueError("Earlier prediction CSV must contain 22 unique drivers.")

    official = pd.DataFrame(
        QUALIFYING,
        columns=["grid_position", "driver_key", "qualifying_position", "q1", "q2", "q3"],
    )
    official["driver_key"] = official["driver_key"].map(surname_key)

    if len(official) != 22 or official["driver_key"].duplicated().any():
        raise ValueError("Official qualifying snapshot has duplicate/missing drivers.")

    missing = set(official["driver_key"]) - set(base["driver_key"])
    unexpected = set(base["driver_key"]) - set(official["driver_key"])
    if missing or unexpected:
        raise ValueError(
            f"Driver mismatch. Missing from base CSV: {missing}; unexpected: {unexpected}"
        )

    # Ensure the earlier source has the same published starting grid.
    old_grid = base.set_index("driver_key")["grid_position"]
    inconsistent = [
        (row.driver_key, int(row.grid_position), old_grid.loc[row.driver_key])
        for row in official.itertuples(index=False)
        if int(old_grid.loc[row.driver_key]) != int(row.grid_position)
    ]
    if inconsistent:
        raise ValueError(
            f"Starting grid differs from prior prediction: {inconsistent}. "
            "Verify the newest FIA grid before continuing."
        )

    # Keep existing driver/team rolling features and source-of-truth FIA grid.
    old_scores = [
        "rank", "raw_score", "win_probability", "win_probability_pct",
        "grid_position", "qualifying_position", "q1_gap_s", "q2_gap_s", "q3_gap_s",
        "has_q2_time", "has_q3_time",
    ]
    base = base.drop(columns=old_scores, errors="ignore")
    df = official.merge(base, on="driver_key", how="left", validate="one_to_one")

    # IMPORTANT: use the same feature-engineering formula as build_qualifying_features.py.
    # Each gap is relative to the best lap within the SAME qualifying segment.
    for segment in ("q1", "q2", "q3"):
        seconds = df[segment].map(lap_to_seconds)
        df[f"{segment}_gap_s"] = seconds - seconds.min(skipna=True)

    df["has_q2_time"] = df["q2"].notna().astype(int)
    df["has_q3_time"] = df["q3"].notna().astype(int)

    saved = joblib.load(MODEL_FILE)  # Only load model files you created/trust.
    features = list(saved["features"])
    model = saved["model"]

    absent_features = [name for name in features if name not in df.columns]
    if absent_features:
        raise ValueError(f"Model expects unavailable columns: {absent_features}")

    for name in features:
        df[name] = pd.to_numeric(df[name], errors="coerce")

    assert len(features) == 13, f"Expected 13 features, found {len(features)}"
    assert df["grid_position"].tolist() == list(range(1, 23))
    assert df["driver"].notna().all()
    assert df["driver"].is_unique

    raw_scores = model.predict_proba(df[features])[:, 1]
    if not np.isfinite(raw_scores).all() or raw_scores.sum() <= 0:
        raise RuntimeError("Model produced invalid scores")

    df["win_probability_pct"] = raw_scores / raw_scores.sum() * 100
    df = df.sort_values("win_probability_pct", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"

    display_cols = [
        "driver", "team", "grid_position", "qualifying_position",
        "q1_gap_s", "q2_gap_s", "q3_gap_s", "win_probability_pct",
    ]
    print("\n2026 SAO PAULO GP — XGBOOST V2 PRE-RACE PREDICTIONS")
    print("FIA qualifying (Nov 7); provisional grid (Nov 7)")
    print(df[display_cols].to_string(
        formatters={"win_probability_pct": "{:.2f}%".format},
        na_rep="—",
    ))
    print(f"\nPredicted winner: {df.iloc[0]['driver']}")
    print(f"Estimated probability: {df.iloc[0]['win_probability_pct']:.2f}%")
    print(f"Probability total: {df['win_probability_pct'].sum():.2f}%")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=True)
    print(f"\nSaved: {OUTPUT_FILE}")
    print("NOTE: Experimental, not yet calibrated probabilities. "
          "Verify any grid changes before the race.")


if __name__ == "__main__":
    main()
