import matplotlib.pyplot as plt

def plot_time_series(data, title="Time Series Plot", xlabel="Time", ylabel="Value"):
    """
    Plots a list of values over time using matplotlib.

    Parameters:
        data (list or array-like): List of values to plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(data, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

hist = []
import csv
def read_in(p):
    with open(p, 'r') as file:
        csvreader = csv.reader(file)
        li = list(csvreader)

        for i in range(1, len(li) - 1):
            mp = float(li[i][0].split(";")[15])
            sec = li[i][0].split(";")[2]

            if sec == "PICNIC_BASKET1":
                hist.append(mp)

# Load all the files
read_in("data/round1/prices_round_1_day_-1.csv")
read_in("data/round1/prices_round_1_day_0.csv")
read_in("data/round1/prices_round_1_day_1.csv")
read_in("cf0d508c-c8d3-4615-b040-37db5e74467f.csv")

plot_time_series(hist)