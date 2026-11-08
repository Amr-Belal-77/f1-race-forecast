
from pathlib import Path

import joblib
import pandas as pd

# 1. Load model and predictions
ROOT = Path(__file__).resolve().parent.parent

model_path = ROOT / "models" / "f1_xgboost_model.joblib"

prediction_path = (
    ROOT / "data" / "processed"
    / "sao_paulo_2026_xgboost_predictions.csv"
)

saved = joblib.load(model_path)

pipeline = saved["model"]
features = saved["features"]

predictions = pd.read_csv(prediction_path)

# 2. Inspect important drivers
drivers = [
    "Andrea Kimi Antonelli",
    "Lando Norris",
    "Max Verstappen",
    "George Russell",
    "Charles Leclerc"
]

selected = predictions[
    predictions["driver"].isin(drivers)
]

columns = [
    "driver",
    "grid_position",
    "qualifying_position",
    "driver_avg_finish_last5",
    "driver_avg_points_last5",
    "driver_win_rate_last5",
    "team_avg_points_last5",
    "win_probability_pct"
]

print("\n===== DRIVER FEATURE COMPARISON =====")
print(selected[columns].to_string(index=False))

# 3. Inspect XGBoost global feature importance
imputer = pipeline.named_steps["imputer"]
classifier = pipeline.named_steps["classifier"]

feature_names = imputer.get_feature_names_out(features)
importances = classifier.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\n===== XGBOOST FEATURE IMPORTANCE =====")
print(importance_df.to_string(index=False))

# 4. Save the analysis
output = (
    ROOT / "data" / "processed"
    / "xgboost_feature_importance.csv"
)

importance_df.to_csv(output, index=False)

print("\nAnalysis saved to:", output)
