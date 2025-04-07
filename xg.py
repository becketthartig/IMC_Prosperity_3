import optuna
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load data
df = pd.read_csv("minor_tools/prices_round_0_day_-1.csv", delimiter=";")

# Feature engineering
df["mid"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
for i in range(1, 6):
    df[f"mid_lag_{i}"] = df["mid"].shift(i)

df["imbalance"] = (
    df["bid_volume_1"] - df["ask_volume_1"]
) / (df["bid_volume_1"] + df["ask_volume_1"])

df["future_mid"] = df["mid"].shift(-3)
df = df.dropna()

features = [col for col in df.columns if "lag" in col or "imbalance" in col]
X = df[features]
y = df["future_mid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Best hyperparameters found by Optuna
best_params = {
    'max_depth': 5,
    'learning_rate': 0.0855176857033861,
    'n_estimators': 450,
    'subsample': 0.9594933258177989,
    'colsample_bytree': 0.9947267062966798,
    'gamma': 0.29886259497929285,
    'min_child_weight': 1
}

# Step 1: Train the final model with the best hyperparameters
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# Step 2: Make predictions using the trained model
y_pred_best = final_model.predict(X_test)

# Evaluate the model with MSE
test_mse = mean_squared_error(y_test, y_pred_best)
print(f"Test MSE with optimized hyperparameters: {test_mse}")

# Optionally, save the trained model for later use
final_model.save_model("best_model.json")
