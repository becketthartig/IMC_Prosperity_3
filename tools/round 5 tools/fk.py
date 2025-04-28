import csv

peeps = ("Caesar", "Camilla", "Paris", "Olivia")
groups = set()
for p in peeps:
    for pp in peeps:
        if p != pp:
            groups.add((p, pp))

products = ("CROISSANTS", "JAMS", "DJEMBES")

grouped_trades_dict = {g: [] for g in groups}

def read_in(p, ts):
    with open(p, 'r') as file:

        csvreader = csv.reader(file)
        li = list(csvreader)

        for i in range(1, len(li)):

            line = li[i][0].split(";")
            product = line[3]

            if product in products:# and ((line[1] == "Camilla" and line[2] == "Paris") or (line[1] == "Paris" and line[2] == "Camilla")):

                grouped_trades_dict[(line[1], line[2])].append((float(line[0]) + ts, float(line[5]), product))

read_in("data/round5/trades_round_5_day_2.csv", 0)
read_in("data/round5/trades_round_5_day_3.csv", 1000000)
read_in("data/round5/trades_round_5_day_4.csv", 2000000)

midprices = {p: [] for p in products}

def read_in2(p, ts):
    with open(p, 'r') as file:

        csvreader = csv.reader(file)
        li = list(csvreader)

        for i in range(1, len(li)):

            line = li[i][0].split(";")
            product = line[2]

            if product in products:
                midprices[product].append((float(line[1]) + ts, float(line[15])))

read_in2("data/round5/prices_round_5_day_2.csv", 0)
read_in2("data/round5/prices_round_5_day_3.csv", 1000000)
read_in2("data/round5/prices_round_5_day_4.csv", 2000000)

import matplotlib.pyplot as plt


colors = ("red", "green", "blue", "orange", "purple", "pink", "yellow", "cyan", "lime", "magenta", "brown", "gray")

group_colors = {g: colors[i] for i, g in enumerate(grouped_trades_dict.keys())}

def plot_stock(stock_name, midprice_data, grouped_trades_dict):
    plt.figure(figsize=(12, 6))
    
    # Plot midprice line
    timestamps, prices = zip(*midprice_data)
    plt.plot(timestamps, prices, label="Midprice", color='cyan', linewidth=0.5)

    # Plot each group's trades for this stock
    for group, trades in grouped_trades_dict.items():
        group_trades = [(ts, price) for ts, price, stock in trades if stock == stock_name]
        if group_trades:
            ts_vals, price_vals = zip(*group_trades)
            plt.scatter(ts_vals, price_vals, label=group, color=group_colors[group], s=25)

    plt.title(f"{stock_name} Midprice and Group Trades")
    plt.xlabel("Timestamp")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot all 3 stocks
for stock in products:
    if stock in midprices:
        plot_stock(stock, midprices[stock], grouped_trades_dict)

