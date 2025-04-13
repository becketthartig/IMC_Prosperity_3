import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_combine_files(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, sep=';')
        df["time_index"] = df["day"] * 1_000_000 + df["timestamp"]
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["mid_price"])
    grouped = combined.groupby(["time_index", "product"])["mid_price"].mean().reset_index()
    return grouped

def plot_strategy(x, y, z, std_window, z_threshold):
    file_paths = [
        "data/round2/prices_round_2_day_-1.csv",
        "data/round2/prices_round_2_day_0.csv",
        "data/round2/prices_round_2_day_1.csv",
        "cf0d508c-c8d3-4615-b040-37db5e74467f.csv"
    ]

    df = load_and_combine_files(file_paths)
    pivoted = df.pivot(index="time_index", columns="product", values="mid_price")

    # Basket price, spread, and std
    basket_price = x * pivoted["CROISSANTS"] + y * pivoted["JAMS"] + z * pivoted["DJEMBES"]
    picnic_price = pivoted["PICNIC_BASKET2"]
    # basket_price2 = ((pivoted["PICNIC_BASKET2"] - 2 * pivoted["JAMS"]) / 4 + (pivoted["PICNIC_BASKET1"] - 3 * pivoted["JAMS"] - pivoted["DJEMBES"]) / 6) / 2
    # picnic_price2 = pivoted["CROISSANTS"]
    basket_price2 = ((pivoted["PICNIC_BASKET2"] - 4 * pivoted["CROISSANTS"]) / 2 + (pivoted["PICNIC_BASKET1"] - 6 * pivoted["CROISSANTS"] - pivoted["DJEMBES"]) / 3) / 2
    picnic_price2 = pivoted["JAMS"]
    spread = picnic_price - basket_price
    spread_std = spread.rolling(window=std_window).std()
    spread2 = picnic_price2 - basket_price2
    spread_std2 = spread2.rolling(window=std_window).std()

    # Z-score with mean = 0
    mean = spread.rolling(std_window).mean()
    z_score = spread2 / spread_std2
    # z_score2 = spread2 / spread_std2
    z_score2 = picnic_price.diff(std_window)
    # Signal logic
    picnic_signal = np.where(z_score > z_threshold, -1,
                      np.where(z_score < -z_threshold, 1, 0))
    basket_signal = -picnic_signal - 3

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                        gridspec_kw={'height_ratios': [1.2, 1.2, 0.6]})

    # Plot 1: Spread and Z-score
    ax1.plot(pivoted.index, spread, label="Spread", color='blue')
    ax1.plot(pivoted.index, spread2, label="Spread", color='orange')
    ax1b = ax1.twinx()
    ax1b.plot(pivoted.index, z_score, label="Z-score", color='purple', linestyle='--')
    # ax1b.plot(pivoted.index, z_score2, label="Z-score", color='green', linestyle='--')
    ax1.axhline(0, color='black', linestyle='--')
    ax1b.axhline(z_threshold, color='red', linestyle='--')
    ax1b.axhline(-z_threshold, color='green', linestyle='--')
    ax1.set_ylabel("Spread")
    ax1b.set_ylabel("Z-score")
    ax1.set_title("Spread and Z-score (Mean = 0)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True)

    # Plot 2: Prices
    ax2.plot(pivoted.index, picnic_price, label="PICNIC_BASKET2", linewidth=2)
    ax2.plot(pivoted.index, basket_price, label=f"Basket2 ({x}C + {y}J + {z}D)", linewidth=2)
    ax2.plot(pivoted.index, picnic_price2, label="PICNIC_BASKET1", linewidth=2)
    ax2.plot(pivoted.index, basket_price2, label=f"Basket1 ({x}C + {y}J + {z}D)", linewidth=2)
    ax2.set_ylabel("Price")
    ax2.set_title("PICNIC_BASKET1 and Basket Prices")
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Signals
    ax3.plot(pivoted.index, picnic_signal, label="PICNIC_BASKET1 Signal", color='orange', drawstyle='steps-post')
    ax3.plot(pivoted.index, basket_signal, label="Basket Signal", color='green', drawstyle='steps-post')
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["Short", "None", "Long"])
    ax3.set_ylabel("Trade Signal")
    ax3.set_xlabel("Time Index")
    ax3.set_title(f"Trade Signals (Z threshold = ±{z_threshold})")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
plot_strategy(x=4, y=2, z=0, std_window=50, z_threshold=10)
