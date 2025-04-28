import pandas as pd
import numpy as np
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler


macaron_prices = []

def read_in(p):
    with open(p, 'r') as file:

        csvreader = csv.reader(file)
        li = list(csvreader)

        for i in range(1, len(li)):

            line = li[i][0].split(";")
            product = line[2]

            if product == "MAGNIFICENT_MACARONS":
                # bid = float(line[3])
                # ask = float(line[9])
                mid_price = float(line[15])

                macaron_prices.append(mid_price)

read_in("data/round4/prices_round_4_day_3.csv")

df = pd.read_csv("data/round4/observations_round_4_day_3.csv")

df = df.drop(columns=["timestamp", "bidPrice", "askPrice"])

df["sugar_lag1"] = df["sugarPrice"].shift(25)
df["sugar_lag2"] = df["sugarPrice"].shift(50)

# Sunlight (longer lag)
df["sun_lag3"] = df["sunlightIndex"].shift(50)
df["sun_lag5"] = df["sunlightIndex"].shift(100)

# Add previous macaron price as input (helps a lot)
# df["prev_macaron_price"] = [np.nan] + macaron_prices[:-1]

# Drop rows with NaNs due to lagging
df = df.dropna()
y = np.array(macaron_prices[len(macaron_prices) - len(df):])  # realign y

# === Train/test split (time-based) ===
split_idx = int(0.8 * len(df))
X_train, X_test = df.iloc[:split_idx], df.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# === Train XGBoost ===
model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# === Predict + Evaluate ===
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)

# === Plot ===
plt.plot(y_test, label="Actual", linewidth=2)
plt.plot(y_pred, label="Predicted", linestyle="--")
plt.legend()
plt.title("Macaron Price Prediction (Lagged Sugar + Sunlight)")
plt.xlabel("Timestep")
plt.ylabel("Price")
plt.show()

plot_importance(model)
plt.show()