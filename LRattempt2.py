import numpy as np
import csv

hist = []

with open('minor_tools/prices_round_0_day_-1.csv', 'r') as file:
    csvreader = csv.reader(file)
    li = list(csvreader)


    for i in range(1, len(li) - 1):
        mp = float(li[i][0].split(";")[15])
        if mp < 9000.0:
            hist.append(mp)
            

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
max_lag = 20

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

print(hist)