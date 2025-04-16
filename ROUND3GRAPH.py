from blackscholes import BLACK_SCHOLES_CALC
from bruh import plot_time_series, plot_time_series2
import csv
import statistics
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

# implied_vols = {s: [] for s in securities}

implied_vol = []

bsmodel = BLACK_SCHOLES_CALC(10000, 7)

# for i, S in enumerate(hists["VOLCANIC_ROCK"]):
#     bsmodel.S = S
#     bsmodel.t = (80000 - i) / 3650000
#     closest = float("inf")
#     closest_k = None
#     for k in strikes:
#         diff = abs(strikes[k] - S)
#         if diff < closest:
#             closest = diff
#             closest_k = k
#     implied_vol.append(bsmodel.implied_vol(strikes[closest_k], hists[closest_k][i]))


# def smooth_implied_vol(implied_vol, window=5):
#     smoothed = []
#     for i in range(len(implied_vol)):
#         if i < window:
#             # Not enough data points yet, just take the current value
#             smoothed.append(implied_vol[i])
#         else:
#             # Average the last 'window' values
#             smoothed.append(sum(implied_vol[i-window:i]) / window)
#     return smoothed



# actual_vol = [0 for _ in range(1000)]
# window = 1000

# for i in range(window - 1, len(hists["VOLCANIC_ROCK"])):
#     window_slice = hists["VOLCANIC_ROCK"][i - window + 1 : i + 1]
#     std_dev = statistics.stdev(window_slice)
#     vol = std_dev / math.sqrt(window)
#     actual_vol.append(vol)


# plot_time_series(implied_vol, "SCATTER OF IMPLIED VOLATILITY OF VOLCANIC_ROCK_VOUCHER_9500")

# prices = hists["VOLCANIC_ROCK"]
# returns = [math.log(prices[i+1] / prices[i]) for i in range(len(prices)-1)]
# volatility = statistics.stdev(returns)
# print(volatility * math.sqrt(3650000))

# plot_time_series(smooth_implied_vol(implied_vol, 15), "implied vol")





plot_time_series(implied_vol, "implied vol")