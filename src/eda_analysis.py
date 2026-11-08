
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load our historical dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent

input_file = (
    PROJECT_ROOT / "data" / "raw"
    / "historical_f1_2018_2026.csv"
)

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)

# 2. Validate our target and race records
race_keys = ["season", "round"]

winners = df.groupby(race_keys)["winner"].sum()

print("\nNumber of historical races:", len(winners))
print("Races with exactly one winner:", (winners == 1).sum())

assert (winners == 1).all()
assert not df.duplicated(
    ["season", "round", "driver_id"]
).any()

# 3. Examine missing qualifying results
print("\nMissing qualifying data:")
print(
    df[
        ["qualifying_position", "q1", "q2", "q3"]
    ].isna().sum()
)

# 4. Check unusual starting positions
print("\nDrivers with grid_position = 0:")
print((df["grid_position"] == 0).sum())

# Only valid positive grid positions for this analysis
valid_grid = df[df["grid_position"] > 0].copy()

# 5. Calculate historical win rate by grid position
grid_stats = (
    valid_grid.groupby("grid_position")
    .agg(
        starts=("winner", "size"),
        wins=("winner", "sum")
    )
    .reset_index()
)

grid_stats["win_rate"] = (
    grid_stats["wins"] / grid_stats["starts"] * 100
)

print("\nWin probability by starting position:")
print(grid_stats.to_string(index=False))

# 6. Identify pole-position win rate
pole = grid_stats[grid_stats["grid_position"] == 1]

if not pole.empty:
    print(
        "\nPole position win rate:",
        round(pole["win_rate"].iloc[0], 2),
        "%"
    )

# 7. Create a visualization
plot_data = grid_stats[
    grid_stats["grid_position"] <= 20
]

plt.figure(figsize=(11, 5))

plt.bar(
    plot_data["grid_position"],
    plot_data["win_rate"]
)

plt.title("F1 Historical Win Rate by Starting Grid Position")
plt.xlabel("Starting Grid Position")
plt.ylabel("Win Rate (%)")
plt.xticks(range(1, 21))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# 8. Save analysis results
output_dir = PROJECT_ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

grid_stats.to_csv(
    output_dir / "grid_win_rates.csv",
    index=False
)

plt.savefig(
    output_dir / "grid_win_rate.png",
    dpi=150
)

plt.show()
