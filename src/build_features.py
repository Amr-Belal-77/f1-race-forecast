
from pathlib import Path

import pandas as pd

# ===================================
# 1. LOAD HISTORICAL DATA
# ===================================

ROOT = Path(__file__).resolve().parent.parent

input_file = (
    ROOT / "data" / "raw"
    / "historical_f1_2018_2026.csv"
)

df = pd.read_csv(input_file)

df["race_date"] = pd.to_datetime(df["race_date"])

df = df.sort_values(
    ["race_date", "season", "round", "driver_id"]
).reset_index(drop=True)


# ===================================
# 2. DRIVER PERFORMANCE FEATURES
# ===================================

# Average finish in previous 5 races
df["driver_avg_finish_last5"] = (
    df.groupby("driver_id")["finish_position"]
    .transform(
        lambda x: x.shift(1).rolling(
            5, min_periods=1
        ).mean()
    )
)

# Average points in previous 5 races
df["driver_avg_points_last5"] = (
    df.groupby("driver_id")["points"]
    .transform(
        lambda x: x.shift(1).rolling(
            5, min_periods=1
        ).mean()
    )
)

# Win rate in previous 5 races
df["driver_win_rate_last5"] = (
    df.groupby("driver_id")["winner"]
    .transform(
        lambda x: x.shift(1).rolling(
            5, min_periods=1
        ).mean()
    )
)

# Historical starts before current race
df["driver_prior_starts"] = (
    df.groupby("driver_id").cumcount()
)


# ===================================
# 3. TEAM PERFORMANCE FEATURES
# ===================================

# First calculate total team points per race
team_df = (
    df.groupby(
        ["season", "round", "race_date", "team_id"],
        as_index=False
    )["points"]
    .sum()
    .rename(columns={"points": "team_race_points"})
)

team_df = team_df.sort_values(
    ["race_date", "season", "round", "team_id"]
)

# Previous 5 team race results ONLY
team_df["team_avg_points_last5"] = (
    team_df.groupby("team_id")["team_race_points"]
    .transform(
        lambda x: x.shift(1).rolling(
            5, min_periods=1
        ).mean()
    )
)

# Join past team performance to driver rows
df = df.merge(
    team_df[
        ["season", "round", "team_id",
         "team_avg_points_last5"]
    ],
    on=["season", "round", "team_id"],
    how="left",
    validate="many_to_one"
)


# ===================================
# 4. SELECT MODEL FEATURES
# ===================================

feature_columns = [
    "grid_position",
    "qualifying_position",
    "driver_avg_finish_last5",
    "driver_avg_points_last5",
    "driver_win_rate_last5",
    "driver_prior_starts",
    "team_avg_points_last5",
]

metadata_columns = [
    "season", "round", "race_date",
    "race_name", "driver_id",
    "driver", "team_id", "team"
]

dataset = df[
    metadata_columns + feature_columns + ["winner"]
].copy()


# ===================================
# 5. SAVE PROCESSED DATA
# ===================================

output_dir = ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "f1_features.csv"

dataset.to_csv(output_file, index=False)


# ===================================
# 6. VERIFY
# ===================================

print("\nFeature dataset shape:", dataset.shape)

print("\nMissing values:")
print(dataset[feature_columns].isna().sum())

print("\nSample of 2026 features:")

sample = dataset[
    dataset["season"] == 2026
].tail(10)

print(sample.to_string(index=False))

print("\nSaved to:", output_file)
