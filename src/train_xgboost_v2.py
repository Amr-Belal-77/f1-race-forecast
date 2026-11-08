from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "f1_features_v2.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "data" / "processed"
CUTOFF_DATE = pd.Timestamp("2026-11-08")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. LOAD DATA AND VALIDATE HISTORICAL RACES
# ==================================================
df = pd.read_csv(DATA_FILE)
df["race_date"] = pd.to_datetime(df["race_date"])
RACE_KEYS = ["season", "round"]
ROW_KEYS = ["season", "round", "driver_id"]

if df.duplicated(ROW_KEYS).any():
    raise ValueError("Duplicate driver-race records detected")
if not (df["race_date"] < CUTOFF_DATE).all():
    raise ValueError("The dataset includes the target race or later races")
if not df.groupby(RACE_KEYS)["winner"].sum().eq(1).all():
    raise ValueError("Each historical race must have exactly one winner")

print("Dataset loaded:", df.shape)

# ==================================================
# 3. CREATE pit_lane_start BEFORE CHECKING FEATURES
# ==================================================
# Jolpica uses grid_position=0 for nonstandard starts.
# This flag does not prove a pit-lane start in every case.
nonstandard = df["grid_position"].isna() | df["grid_position"].le(0)
df["pit_lane_start"] = nonstandard.astype(int)
max_grid = df.groupby(RACE_KEYS)["grid_position"].transform("max")
df.loc[nonstandard, "grid_position"] = max_grid[nonstandard] + 1
print("Nonstandard grid positions:", int(nonstandard.sum()))

# ==================================================
# 4. DEFINE V1 AND V2 FEATURES
# ==================================================
old_model_path = MODEL_DIR / "f1_xgboost_model.joblib"
v1_saved = joblib.load(old_model_path)
V1_FEATURES = list(v1_saved["features"])

NEW_FEATURES = [
    "q1_gap_s", "q2_gap_s", "q3_gap_s",
    "has_q2_time", "has_q3_time",
]
V2_FEATURES = V1_FEATURES + NEW_FEATURES

if len(set(V2_FEATURES)) != len(V2_FEATURES):
    raise ValueError("The feature list contains duplicate columns")
missing_columns = [column for column in V2_FEATURES if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing feature columns: {missing_columns}")

print("V1 feature count:", len(V1_FEATURES))
print("V2 feature count:", len(V2_FEATURES))
print("\nQualifying missing values:")
print(df[NEW_FEATURES].isna().sum())

# ==================================================
# 5. CHRONOLOGICAL SPLIT
# ==================================================
train = df[df["season"] <= 2024].copy()
validation = df[df["season"] == 2025].copy()
diagnostic = df[df["season"] == 2026].copy()

if train.empty or validation.empty or diagnostic.empty:
    raise ValueError("Training, validation, or 2026 diagnostic set is empty")
print("\nTraining rows:", len(train))
print("Validation rows:", len(validation))
print("2026 diagnostic rows:", len(diagnostic))

# ==================================================
# 6. MODEL BUILDER
# ==================================================
def create_model():
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

# ==================================================
# 7. EVALUATE ONE RACE AT A TIME
# ==================================================
def evaluate(model, data, features):
    result = data[["season", "round", "driver", "grid_position", "winner"]].copy()
    result["raw_score"] = model.predict_proba(data[features])[:, 1]
    total = result.groupby(RACE_KEYS)["raw_score"].transform("sum")
    if not (total > 0).all():
        raise ValueError("Invalid zero-sum race prediction")
    result["probability"] = result["raw_score"] / total

    chosen = result.loc[result.groupby(RACE_KEYS)["probability"].idxmax()]
    pole = result.loc[result.groupby(RACE_KEYS)["grid_position"].idxmin()]
    winners = result.loc[result["winner"].eq(1), "probability"]

    squared = (result["probability"] - result["winner"]) ** 2
    brier = result.assign(squared_error=squared).groupby(RACE_KEYS)["squared_error"].sum().mean()

    return {
        "races": len(chosen),
        "accuracy": chosen["winner"].mean(),
        "pole_accuracy": pole["winner"].mean(),
        "log_loss": -np.log(winners.clip(lower=1e-12)).mean(),
        "brier": brier,
    }

# ==================================================
# 8. CONTROLLED COMPARISON
# ==================================================
all_results = []
for name, features in [("XGBoost_V1", V1_FEATURES), ("XGBoost_V2", V2_FEATURES)]:
    print(f"\n========== {name} ==========")
    model = create_model()
    model.fit(train[features], train["winner"])

    for season_name, subset in [("2025", validation), ("2026", diagnostic)]:
        metrics = evaluate(model, subset, features)
        all_results.append({"model": name, "season": season_name, **metrics})
        print(f"\n{season_name} ({metrics['races']} races)")
        print(f"Top-1 Accuracy: {metrics['accuracy']:.2%}")
        print(f"Pole Baseline: {metrics['pole_accuracy']:.2%}")
        print(f"Race Log Loss: {metrics['log_loss']:.4f}")
        print(f"Brier Score: {metrics['brier']:.4f}")

comparison = pd.DataFrame(all_results)
comparison_file = OUTPUT_DIR / "xgboost_v1_vs_v2.csv"
comparison.to_csv(comparison_file, index=False)
print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))
print("\nComparison saved to:", comparison_file)

# ==================================================
# 9. RETRAIN V2 FOR THE UPCOMING RACE
# ==================================================
final_model = create_model()
final_model.fit(df[V2_FEATURES], df["winner"])
model_file = MODEL_DIR / "f1_xgboost_v2_model.joblib"
joblib.dump({
    "model": final_model,
    "features": V2_FEATURES,
    "version": "V2_qualifying_pace",
}, model_file)
print("\nV2 model saved to:", model_file)
print("IMPORTANT: Today's prediction script still needs the five new qualifying features.")
