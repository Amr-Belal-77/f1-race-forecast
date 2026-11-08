
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================
# 1. LOAD DATASETS
# =====================================

ROOT = Path(__file__).resolve().parent.parent

raw = pd.read_csv(
    ROOT / "data" / "raw"
    / "historical_f1_2018_2026.csv"
)

features = pd.read_csv(
    ROOT / "data" / "processed"
    / "f1_features.csv"
)

KEYS = ["season", "round", "driver_id"]

assert not raw.duplicated(KEYS).any()
assert not features.duplicated(KEYS).any()


# =====================================
# 2. CONVERT LAP TIMES TO SECONDS
# =====================================

def lap_to_seconds(value):
    """
    Convert '1:27.470' into 87.470 seconds.

    Missing times remain NaN.
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value in ("", "-", "\\N"):
        return np.nan

    parts = value.split(":")

    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    if len(parts) == 1:
        return float(parts[0])

    raise ValueError(f"Unexpected lap time: {value}")


# =====================================
# 3. MERGE QUALIFYING TIMES
# =====================================

df = features.merge(
    raw[KEYS + ["q1", "q2", "q3"]],
    on=KEYS,
    how="left",
    validate="one_to_one"
)

assert len(df) == len(features)


# =====================================
# 4. CALCULATE LAP-TIME GAPS
# =====================================

RACE_KEYS = ["season", "round"]

for session in ["q1", "q2", "q3"]:

    seconds_column = f"{session}_seconds"
    gap_column = f"{session}_gap_s"

    # Convert times to seconds
    df[seconds_column] = df[session].apply(
        lap_to_seconds
    )

    # Fastest recorded lap in the SAME session
    fastest = df.groupby(
        RACE_KEYS
    )[seconds_column].transform("min")

    # Driver's time difference from session best
    df[gap_column] = (
        df[seconds_column] - fastest
    ).clip(lower=0)


# =====================================
# 5. QUALIFYING SESSION INDICATORS
# =====================================

df["has_q2_time"] = (
    df["q2_seconds"].notna().astype(int)
)

df["has_q3_time"] = (
    df["q3_seconds"].notna().astype(int)
)


# =====================================
# 6. SELECT NEW FEATURES
# =====================================

NEW_FEATURES = [
    "q1_gap_s",
    "q2_gap_s",
    "q3_gap_s",
    "has_q2_time",
    "has_q3_time"
]

# Remove raw timing columns from ML dataset
df = df.drop(
    columns=[
        "q1", "q2", "q3",
        "q1_seconds", "q2_seconds", "q3_seconds"
    ]
)


# =====================================
# 7. DATA VALIDATION
# =====================================

assert not df.duplicated(KEYS).any()

for column in ["q1_gap_s", "q2_gap_s", "q3_gap_s"]:

    nonmissing = df[column].dropna()

    assert (nonmissing >= 0).all()

assert df["winner"].notna().all()

print("\nNew qualifying features:")
print(df[NEW_FEATURES].describe().to_string())

print("\nMissing values:")
print(df[NEW_FEATURES].isna().sum())

print("\nHistorical records:", len(df))

print("\nLatest 2026 historical qualifying features:")
print(
    df[df["season"] == 2026][
        ["driver", "qualifying_position"] + NEW_FEATURES
    ].tail(10).to_string(index=False)
)


# =====================================
# 8. SAVE NEW FEATURE DATASET
# =====================================

output = (
    ROOT / "data" / "processed"
    / "f1_features_v2.csv"
)

df.to_csv(output, index=False)

print("\nSaved to:", output)
