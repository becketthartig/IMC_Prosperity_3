import numpy as np
import csv
import matplotlib.pyplot as plt

hist = []

def read_in(p):
    with open(p, 'r') as file:
        csvreader = csv.reader(file)
        li = list(csvreader)

        for i in range(1, len(li) - 1):
            mp = float(li[i][0].split(";")[15])
            sec = li[i][0].split(";")[2]

            if sec == "SQUID_INK":
                hist.append(mp)

# Load all the files
read_in("data/round1/prices_round_1_day_-2.csv")
read_in("data/round1/prices_round_1_day_-1.csv")
read_in("data/round1/prices_round_1_day_0.csv")
read_in("data/round1/prices_round_1_day_1.csv")

window = 250
momentum_window = 200  # Customize as needed
prices = np.array(hist)

# Rolling mean and std deviation
rolling_mean = np.zeros_like(prices)
rolling_std = np.zeros_like(prices)
z_scores = np.zeros_like(prices)

for i in range(window, len(prices)):
    window_slice = prices[i - window:i]
    mean = np.mean(window_slice)
    std = np.std(window_slice)
    rolling_mean[i] = mean
    rolling_std[i] = std
    if std > 0:
        z_scores[i] = (prices[i] - mean) / std

# Momentum calculation
momentum = np.zeros_like(prices)
for i in range(momentum_window, len(prices)):
    momentum[i] = prices[i] - prices[i - momentum_window]

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

# Mid price chart
axs[0].plot(prices, label="Mid Price", color='blue')
axs[0].plot(rolling_mean, label="Rolling Mean (200)", color='green', alpha=0.7)
axs[0].set_title("Mid Price with Rolling Mean")
axs[0].set_ylabel("Price")
axs[0].legend()

# Z-score chart
axs[1].plot(z_scores, label="Z-Score (window=200)", color='orange')
axs[1].axhline(2, color='red', linestyle='--', label='Z = ±2')
axs[1].axhline(-2, color='red', linestyle='--')
axs[1].set_title("Z-Score of Mid Price (Mean Reversion Signal)")
axs[1].set_ylabel("Z-Score")
axs[1].legend()

# Momentum chart
axs[2].plot(momentum, label=f"Momentum ({momentum_window}-tick)", color='purple')
axs[2].axhline(0, color='gray', linestyle='--')
axs[2].set_title("Momentum of Mid Price")
axs[2].set_ylabel("Momentum")
axs[2].set_xlabel("Timestep")
axs[2].legend()

plt.tight_layout()
plt.show()
