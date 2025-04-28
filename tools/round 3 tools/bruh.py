import matplotlib.pyplot as plt

def plot_time_series(data, title="Time Series Plot", xlabel="Time", ylabel="Value"):
    plt.figure(figsize=(10, 4))
    plt.plot(data, color='red', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_time_series2(data, data2, title="Time Series Plot", xlabel="Time", ylabel="Value"):
    plt.figure(figsize=(10, 4))
    plt.plot(data, color='red', linestyle='-')
    plt.plot(data2, color='blue', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_time_series4(data, data2, data3, data4, title="Time Series Plot", xlabel="Time", ylabel="Value"):
    plt.figure(figsize=(10, 4))
    plt.plot(data, color='red', linestyle='-', label="bid")
    plt.plot(data2, color='blue', linestyle='-', label="ask")
    plt.plot(data3, color='orange', linestyle='-', label="imp_bid")
    plt.plot(data4, color='green', linestyle='-', label="imp_ask")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# hist = []
# import csv
# def read_in(p):
#     with open(p, 'r') as file:
#         csvreader = csv.reader(file)
#         li = list(csvreader)

#         for i in range(1, len(li) - 1):
#             mp = float(li[i][0].split(";")[15])
#             sec = li[i][0].split(";")[2]

#             if sec == "PICNIC_BASKET1":
#                 hist.append(mp)

# # Load all the files
# read_in("data/round2/prices_round_2_day_-1.csv")
# read_in("data/round2/prices_round_2_day_0.csv")
# read_in("data/round2/prices_round_2_day_1.csv")
# read_in("cf0d508c-c8d3-4615-b040-37db5e74467f.csv")
# read_in("data/round3/prices_round_3_day_2.csv")

# plot_time_series(hist)