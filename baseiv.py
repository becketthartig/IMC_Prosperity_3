import numpy as np
import matplotlib.pyplot as plt
import math

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
        import csv
        csvreader = csv.reader(file)
        li = list(csvreader)

        for i in range(1, len(li)):
            line = li[i][0].split(";")
            product = line[2]
            mid_price = float(line[15])

            if product in securities:
                hists[product].append(mid_price)

read_in("data/round3/prices_round_3_day_0.csv")
read_in("data/round3/prices_round_3_day_1.csv")
read_in("data/round3/prices_round_3_day_2.csv")

base_ivs = []
timestamps = []

for i, S in enumerate(hists["VOLCANIC_ROCK"]):
    tte = (80000 - i) / 3650000
    if tte <= 0: continue

    m_list = []
    iv_list = []

    for voucher, K in strikes.items():
        if i >= len(hists[voucher]): continue
        V = hists[voucher][i]
        from blackscholes import BLACK_SCHOLES_CALC
        bsmodel = BLACK_SCHOLES_CALC(S, tte * 365)

        try:
            implied = bsmodel.implied_vol(K, V)
        except:
            continue

        m = math.log(K / S) / math.sqrt(tte)
        m_list.append(m)
        iv_list.append(implied)

    if len(m_list) < 3:  # not enough points to fit a parabola
        continue

    # Fit parabola: v(m) = a m^2 + b m + c
    # print(len(m_list))
    m_array = np.array(m_list)
    iv_array = np.array(iv_list)
    coeffs = np.polyfit(m_array, iv_array, 2)
    a, b, c = coeffs

    base_ivs.append(c)  # v(m=0) = c
    timestamps.append(i)

# Plot base IV over time
plt.figure(figsize=(9, 4))
plt.plot(timestamps, base_ivs)
plt.xlabel("Time (ticks)")
plt.ylabel("Base Implied Volatility (ATM)")
plt.title("Base IV (v(m=0)) Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()