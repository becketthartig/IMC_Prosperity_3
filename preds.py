import xgboost as xgb
import pandas as pd

# Load the saved model
loaded_model = xgb.XGBRegressor()
loaded_model.load_model("best_model.json")

# Load new data for prediction (assuming it's the same format as the training data)
# Example: load new data from a CSV or other sources
new_data = pd.read_csv("minor_tools/prices_round_0_day_-1.csv", delimiter=";")

# Feature engineering on new data (same as what was done during training)
new_data["mid"] = (new_data["bid_price_1"] + new_data["ask_price_1"]) / 2
for i in range(1, 6):
    new_data[f"mid_lag_{i}"] = new_data["mid"].shift(i)

new_data["imbalance"] = (
    new_data["bid_volume_1"] - new_data["ask_volume_1"]
) / (new_data["bid_volume_1"] + new_data["ask_volume_1"])

# Drop the NaN values from the new data
new_data = new_data.dropna()

# Define the features to be used for prediction (same as during training)
features = [col for col in new_data.columns if "lag" in col or "imbalance" in col]
X_new = new_data[features]

# Make predictions with the loaded model
y_pred_new = loaded_model.predict(X_new)

# Optionally, you can save or print the predictions
print("Predictions:", y_pred_new)