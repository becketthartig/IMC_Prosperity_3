import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts

df = pd.read_csv("/Users/becketthartig/dev/P3 - Literal Zero/minor tools/c64baf0e-bf54-4a56-b807-51139d48fab0.csv", delimiter=';')

# Filter rows where 'product' is 'RAINFOREST_RESIN'
filtered_df = df[df["product"] == "RAINFOREST_RESIN"]

# Extract the relevant column as time series (replace 'your_column' with actual column name)
X = filtered_df["mid_price"].to_numpy()

# Identify deviation points (crossing above 10,000.5 or below 9,999.5)
threshold_high, threshold_low = 10002.5, 10001.5
deviation_points = np.where((X > threshold_high) | (X < threshold_low))[0]

# Define step size N (how far ahead to check movement)
N = 5
returns = np.array([X[t+N] - X[t] for t in deviation_points if t+N < len(X)])

# Compute lag-1 autocorrelation
lag = 2
autocorr = ts.acf(returns, nlags=lag)[lag]
print(f"Lag-{lag} autocorrelation: {autocorr}")

# Interpretation:
if autocorr > 0:
    print("Momentum detected: The variable tends to keep moving in the same direction after crossing 10,000.")
elif autocorr < 0:
    print("Mean reversion detected: The variable tends to reverse direction after crossing 10,000.")
else:
    print("No significant pattern detected.")