import pandas as pd
import matplotlib.pyplot as plt

def plot_csv(file_path, x_column, y_column):
    # Load CSV file with semicolon delimiter
    df = pd.read_csv(file_path, delimiter=';')
    
    # Filter rows where 'product' is 'RAINFOREST_RESIN'
    df = df[df['product'] == 'RAINFOREST_RESIN']
    df = df[df['timestamp'] < 20000]
    # df = df[df['product'] == 'KELP']
    
    # Sort by x_column in case it's unordered
    df = df.sort_values(by=x_column)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df[x_column], df[y_column], marker='o', markersize=2, linestyle='-')
    
    # Labels and title
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f'{y_column} vs {x_column}')
    
    # Show the plot
    plt.show()

# Example usage:
plot_csv('/Users/becketthartig/dev/P3 - Literal Zero/minor_tools/prices_round_0_day_-1.csv', 'timestamp', 'ask_price_1')
