import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_csv_with_slope_and_derivative(file_path, x_column, y_column, lookback=13):
    # Load CSV file with semicolon delimiter
    df = pd.read_csv(file_path, delimiter=';')

    # Filter rows where 'product' is 'KELP'
    df = df[df['product'] == 'KELP']
    
    # Sort by x_column in case it's unordered
    df = df.sort_values(by=x_column)

    # Compute slope of the best-fit line for each time step
    slopes = [np.nan] * lookback  # First 'lookback' values have no slope
    
    for i in range(lookback, len(df)):
        # Extract the past 'lookback' time steps
        x_values = np.array(range(lookback)).reshape(-1, 1)  # Time step indices
        y_values = df[y_column].iloc[i-lookback:i].values.reshape(-1, 1)  # Past prices

        # Fit a linear regression model
        model = LinearRegression().fit(x_values, y_values)
        slopes.append(model.coef_[0][0])  # Extract the slope

    # Add slope values to the dataframe
    df["slope"] = slopes

    # Compute the rate of change of the slope (difference between consecutive slopes)
    df["slope_rate_of_change"] = df["slope"].diff()

    # Create three side-by-side subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot mid_price vs. timestamp
    axes[0].plot(df[x_column], df[y_column], marker='o', markersize=2, linestyle='-')
    axes[0].set_xlabel(x_column)
    axes[0].set_ylabel(y_column)
    axes[0].set_title(f'{y_column} vs {x_column}')

    # Plot slope vs. timestamp
    axes[1].plot(df[x_column], df["slope"], marker='o', markersize=2, linestyle='-', color='red')
    axes[1].set_xlabel(x_column)
    axes[1].set_ylabel("Slope of Best-Fit Line")
    axes[1].set_title("Slope Over Time")

    # Plot rate of change of the slope vs. timestamp
    axes[2].plot(df[x_column], df["slope_rate_of_change"], marker='o', markersize=2, linestyle='-', color='green')
    axes[2].set_xlabel(x_column)
    axes[2].set_ylabel("Rate of Change of Slope")
    axes[2].set_title("Rate of Change of Slope Over Time")

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

# Example usage:
plot_csv_with_slope_and_derivative('/Users/becketthartig/dev/P3 - Literal Zero/minor_tools/prices_round_0_day_-1.csv', 'timestamp', 'mid_price')
