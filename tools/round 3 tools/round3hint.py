from blackscholes import BLACK_SCHOLES_CALC
import csv
import matplotlib.pyplot as plt
import math
import bruh

securities = ("VOLCANIC_ROCK",
              "VOLCANIC_ROCK_VOUCHER_9500",
              "VOLCANIC_ROCK_VOUCHER_9750",
              "VOLCANIC_ROCK_VOUCHER_10000",
              "VOLCANIC_ROCK_VOUCHER_10250",
              "VOLCANIC_ROCK_VOUCHER_10500")
strikes = {"VOLCANIC_ROCK_VOUCHER_9500": 9500,
           "VOLCANIC_ROCK_VOUCHER_9750": 9750,
           "VOLCANIC_ROCK_VOUCHER_10000": 10000,
           "VOLCANIC_ROCK_VOUCHER_10250": 10250,
           "VOLCANIC_ROCK_VOUCHER_10500": 10500}

hists = {s: [] for s in securities}

def read_in(p):
    with open(p, 'r') as file:

        csvreader = csv.reader(file)
        li = list(csvreader)

        for i in range(1, len(li)):

            line = li[i][0].split(";")
            product = line[2]
            mid_price = float(line[15])

            if product in securities:
                hists[product].append(mid_price)



# read_in("data/round3/prices_round_3_day_0.csv")
# read_in("data/round3/prices_round_3_day_1.csv")
read_in("data/round3/prices_round_3_day_2.csv")

graphing = 9500
graphingstr = "VOLCANIC_ROCK_VOUCHER_" + str(graphing)
m_t = []
implied_vols = []
expected_vols = []
bsmodel = BLACK_SCHOLES_CALC(10000, 7)

diff = []

for i, S in enumerate(hists["VOLCANIC_ROCK"]):
    bsmodel.S = S
    bsmodel.t = (60000 - i) / 3650000
    for k in strikes:
        if k == "VOLCANIC_ROCK_VOUCHER_10250" or True:
            iv = bsmodel.implied_vol(strikes[k], hists[k][i])
            if iv < 0.13:
                continue
            mt = math.log(strikes[k] / S) / math.sqrt(bsmodel.t)
            # if mt < -0.47:
            #     continue
            m_t.append(mt)
            implied_vols.append(iv)
            # expected_vols.append(0.23938 * mt ** 2 + 0.00256 * mt + 0.14828)
            expected_vols.append(0.23729 * mt ** 2 + 0.00294 * mt + 0.14920)
            diff.append(implied_vols[-1] - expected_vols[-1])

# bruh.plot_time_series(diff)
# for i in range(1):
#     m_t.append(0)
#     implied_vols.append(0.15)

plt.figure(figsize=(8, 4))
plt.plot(m_t, implied_vols, linestyle='None', marker='o')
plt.xlabel('moneyness values')
plt.ylabel('implied vol values')
plt.title('moneyness vs implied vol')
plt.grid(True)

import numpy as np

# Convert your m_t and implied_vols to numpy arrays
m_array = np.array(m_t)
iv_array = np.array(implied_vols)

# Fit a 2nd degree polynomial (parabola)
coeffs = np.polyfit(m_array, iv_array, deg=2)  # returns [a, b, c]
a, b, c = coeffs

# Print the equation
print(f"Fitted Parabola: v(m) = {a:.5f} * m^2 + {b:.5f} * m + {c:.5f}")

# Create smooth m values for plotting the curve
m_fit = np.linspace(min(m_array), max(m_array), 300)
v_fit = a * m_fit**2 + b * m_fit + c

# Plot the fitted parabola
plt.plot(m_fit, v_fit, color='red', label='Fitted parabola')

# Add legend and show
plt.legend()

plt.show()

