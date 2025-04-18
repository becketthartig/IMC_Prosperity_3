import csv
import bruh

securities = ("MAGNIFICENT_MACARONS")

bids = []
asks = []

def read_in(p):
    with open(p, 'r') as file:

        csvreader = csv.reader(file)
        li = list(csvreader)

        for i in range(1, len(li)):

            line = li[i][0].split(";")
            product = line[2]

            if product == "MAGNIFICENT_MACARONS":
                bid = float(line[3])
                ask = float(line[9])
            # mid_price = float(line[15])

                bids.append(bid)
                asks.append(ask)

implied_bids = []
implied_asks = []
sugs = []
suns = []

def read_in_implied(p):
    with open(p, 'r') as file:

        csvreader = csv.reader(file)
        li = list(csvreader)

        for i in range(1, len(li)):

            line = li[i]
            bid = float(line[1])
            ask = float(line[2])
            trans = float(line[3])
            export = float(line[4])
            imp = float(line[5])
            sug = (float(line[6]) * 5 - 600) 
            sun = float(line[7])  * 4

            implied_bids.append(bid - export - trans)
            implied_asks.append(ask + imp + trans)
            sugs.append(sug)
            suns.append(sun)

read_in("data/round4/prices_round_4_day_1.csv")
read_in("data/round4/prices_round_4_day_2.csv")
read_in("data/round4/prices_round_4_day_3.csv")
read_in("r4.csv")

read_in_implied("data/round4/observations_round_4_day_1.csv")
read_in_implied("data/round4/observations_round_4_day_2.csv")
read_in_implied("data/round4/observations_round_4_day_3.csv")


# bruh.plot_time_series4(bids, asks, implied_bids, implied_asks)
bruh.plot_time_series4(bids, asks, sugs, suns)

            