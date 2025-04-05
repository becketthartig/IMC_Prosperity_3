import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd

df = pd.read_csv("minor_tools/prices_round_0_day_-1.csv", delimiter=";")

# Create features
df["mid"] = (df["bid_price_1"] + df["ask_price_1"]) / 2
for i in range(1, 6):
    df[f"mid_lag_{i}"] = df["mid"].shift(i)

df["imbalance"] = (
    df["bid_volume"] - df["ask_volume"]
) / (df["bid_volume"] + df["ask_volume"])

# Label: future mid price
df["future_mid"] = df["mid"].shift(-3)
df = df.dropna()

features = [col for col in df.columns if "lag" in col or "imbalance" in col]
X = df[features]
y = df["future_mid"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
model.fit(X_train, y_train)

print("Test MSE:", mean_squared_error(y_test, model.predict(X_test)))

# Save for later inference
model.save_model("kelp_model.json")