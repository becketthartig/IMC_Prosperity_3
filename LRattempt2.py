import numpy as np
import csv

hist = []

def read_in(p):
    with open(p, 'r') as file:
        csvreader = csv.reader(file)
        li = list(csvreader)


        for i in range(1, len(li) - 1):
            mp = float(li[i][0].split(";")[15])
            sec = li[i][0].split(";")[2]

            if sec == "KELP":
                hist.append(mp)

read_in("round-1-island-data-bottle/prices_round_1_day_-2.csv")
read_in("round-1-island-data-bottle/prices_round_1_day_-1.csv")
read_in("round-1-island-data-bottle/prices_round_1_day_0.csv")
            

# Example list of historical prices (replace this with actual game data)
p = np.array(hist)

def train_and_evaluate(prices, max_lag, test_ratio=0.2):
    """
    Finds the optimal lag value while checking for overfitting.

    Parameters:
    - prices: List or NumPy array of historical prices.
    - max_lag: Maximum number of lags to test.
    - test_ratio: Fraction of data to use for testing.

    Returns:
    - Dictionary of {lag: (train MSE, test MSE)}
    - Best lag based on test MSE.
    - Best model coefficients (theta) for the optimal lag.
    """
    if len(prices) <= max_lag:
        raise ValueError("Not enough data points for the chosen max lag.")

    mse_results = {}
    best_theta = None
    best_lag = None  # Initialize best_lag

    # Split data into training and testing sets
    split_index = int(len(prices) * (1 - test_ratio))
    train_prices, test_prices = prices[:split_index], prices[split_index:]

    for lag in range(1, max_lag + 1):
        # Prepare training data
        X_train, y_train = [], []
        for i in range(len(train_prices) - lag):
            X_train.append(train_prices[i:i + lag])
            y_train.append(train_prices[i + lag])

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]  # Bias term

        # Solve for theta using Normal Equation
        theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        # Compute train MSE
        y_train_pred = X_train @ theta
        train_mse = np.mean((y_train - y_train_pred) ** 2)

        # Prepare test data
        X_test, y_test = [], []
        for i in range(len(test_prices) - lag):
            X_test.append(test_prices[i:i + lag])
            y_test.append(test_prices[i + lag])

        if len(X_test) == 0:  # If not enough test data for this lag, skip it
            continue

        X_test = np.array(X_test)
        y_test = np.array(y_test)
        X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # Bias term

        # Compute test MSE
        y_test_pred = X_test @ theta
        test_mse = np.mean((y_test - y_test_pred) ** 2)

        mse_results[lag] = (train_mse, test_mse)

        # Store the best model's coefficients
        if best_theta is None or test_mse < mse_results.get(best_lag, (float('inf'), float('inf')))[1]:
            best_theta = theta
            best_lag = lag  # Assign best lag here

    # Ensure best_lag is assigned before returning
    if best_lag is None:
        raise ValueError("No valid lag found. Check if you have enough data.")

    return mse_results, best_lag, best_theta

# Example Usage
# prices = np.array([100.1, 100.2, 100.15, 100.25, 100.3, 100.35, 100.4, 100.45, 100.5, 100.55, 100.6, 100.65])
max_lag = 4

mse_results, best_lag, best_theta = train_and_evaluate(p, max_lag)

print("MSE Results for Different Lags:")
for lag, (train_mse, test_mse) in mse_results.items():
    print(f"Lag {lag}: Train MSE = {train_mse:.5f}, Test MSE = {test_mse:.5f}")

print(f"\nOptimal Lag: {best_lag} (Lowest Test MSE = {mse_results[best_lag][1]:.5f})")

# Print the equation for the best lag
equation = f"Predicted Price = {best_theta[0]:.5f}"
for i in range(1, len(best_theta)):
    equation += f" + ({best_theta[i]:.5f} * Price_{i})"
print("\nBest Linear Regression Equation:")
print(equation)


import matplotlib.pyplot as plt

def plot_lists(actual, predicted, p2, title="Actual vs Predicted Prices"):
    """
    Plots two lists for comparison.
    
    Parameters:
    - actual: List of actual values (ground truth)
    - predicted: List of predicted values (model output)
    - title: Title of the plot (optional)
    """
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual", marker='s', linestyle="-")
    plt.plot(predicted, label="momentum", linestyle="-")
    plt.plot(p2, label="slope", linestyle="-")
    
    plt.xlabel("Time / Index")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()




predicted_prices = [0, 0, 0, 0]
predicted_prices2 = [0, 0, 0, 0]

for i in range(3, len(hist) - 1):
    predicted_prices.append(18.40810 + (0.16527 * hist[i-3]) + (0.14608 * hist[i-2]) + (0.27305 * hist[i-1]) + (0.40648 * hist[i]))
    predicted_prices2.append(18.40810 + (0.16527 * hist[i-2]) + (0.14608 * hist[i-1]) + (0.27305 * hist[i]) + (0.40648 * predicted_prices[i+1]))

def smooth_price(hist, alpha=0.2):
    smoothed = [hist[0]]
    for p in hist[1:]:
        smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
    return smoothed

def mean_squared_error(actual, predicted):
    """
    Computes the Mean Squared Error (MSE) between two lists or NumPy arrays.

    Parameters:
    - actual: List or NumPy array of actual values.
    - predicted: List or NumPy array of predicted values.

    Returns:
    - MSE (float)
    """
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean((actual - predicted) ** 2)

print(mean_squared_error(hist[4:], predicted_prices[4:]))
print(mean_squared_error(hist[4:], predicted_prices2[4:]))

# plot_lists(smooth_price(hist), predicted_prices, predicted_prices2)

import numpy as np

def compute_momentum_series(hist, window=5, method=1, alpha=0.2, apply_volatility_filter=True):
    """
    Computes a momentum signal series for the given historical price list.
    
    Parameters:
        hist (list of float): Historical price data.
        window (int): Window size to calculate momentum.
        method (int): 
            1 = Simple diff mean
            2 = EMA-based momentum
            3 = Rolling linear regression slope
            4 = Filtered momentum (standalone)
            5 = Cumulative smoothed momentum
        alpha (float): EMA smoothing factor (used in method 2 and 5).
        apply_volatility_filter (bool): Whether to ignore low-signal momentum based on volatility.

    Returns:
        list of float: Momentum values per point in history.
    """
    n = len(hist)
    momentum_series = []
    cumulative_buffer = []  # for method 5

    def ema(prices):
        ema_val = prices[0]
        for p in prices[1:]:
            ema_val = alpha * p + (1 - alpha) * ema_val
        return ema_val

    for i in range(n):
        if i < window:
            momentum_series.append(0)
            continue

        window_prices = hist[i-window:i]

        if method == 1:
            mom = np.mean(np.diff(window_prices))

        elif method == 2:
            ema_now = ema(window_prices)
            ema_prev = ema(window_prices[:-1])
            mom = ema_now - ema_prev

        elif method == 3:
            y = np.array(window_prices)
            x = np.arange(window)
            A = np.vstack([x, np.ones_like(x)]).T
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            mom = slope

        elif method == 4:
            mom = np.mean(np.diff(window_prices))
            volatility = np.std(window_prices)
            mom = mom if abs(mom) > 0.35 * volatility else 0

        elif method == 5:
            # Cumulative average of momentum deltas
            recent_mom = np.mean(np.diff(window_prices))
            cumulative_buffer.append(recent_mom)
            if len(cumulative_buffer) > 20:
                cumulative_buffer.pop(0)
            mom = np.mean(cumulative_buffer)

        else:
            raise ValueError("Unsupported method. Use 1–5.")

        # Apply volatility filter by default (except method 4 which already does)
        if apply_volatility_filter and method != 4:
            volatility = np.std(window_prices)
            if abs(mom) < 0.5 * volatility:
                mom = 0

        momentum_series.append(mom)

    # return momentum_series


    return [m * 20 for m in momentum_series]



hists = smooth_price(hist, 0.2)

m1 = compute_momentum_series(hists, window=8, method=4, alpha=0.2, apply_volatility_filter=False)
# m2 = compute_momentum_series(hists, window=15, method=3)
# m3 = compute_momentum_series(hists, window=30, method=3)
# combo = [m1[i] + m2[i] + m3[i] for i in range(len(hists))]

plot_lists([(h - 2000) * 2 for h in hist], m1, [1.4 for _ in range(len(hists))])

