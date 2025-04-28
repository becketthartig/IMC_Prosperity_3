import csv
import math

hist = []

def read_in(p):
    with open(p, 'r') as file:
        csvreader = csv.reader(file)
        li = list(csvreader)


        for i in range(1, len(li) - 1):
            mp = float(li[i][0].split(";")[15])
            sec = li[i][0].split(";")[2]

            if sec == "KELP":
                hist.append(mp)

read_in("data/round3/prices_round_3_day_1.csv")


def calculate_variance(data, sample=True):
    if not data:
        return None  # Handle empty list

    n = len(data)
    mean = sum(data) / n
    squared_diffs = [(x - mean) ** 2 for x in data]

    if sample:
        return sum(squared_diffs) / (n - 1)
    else:
        return sum(squared_diffs) / n

print(len(hist))
# Example usage:
data = []
for i in range(len(hist)):
    if i % 334 == 0:
        data.append(hist[i])


# print("Sample variance:", calculate_variance(data_points))
print("Population variance:", calculate_variance(data, sample=False))
print(len(data))
print(calculate_variance(data, sample=False) / math.sqrt(len(data)))
