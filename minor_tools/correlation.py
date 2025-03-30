import csv

with open('/Users/becketthartig/dev/P3 - Literal Zero/minor_tools/prices_round_0_day_-1.csv', 'r') as file:
    csvreader = csv.reader(file)
    li = list(csvreader)


    sm = 0
    div = 0

    for i in range(1, len(li) - 1):
        if float(li[i][0].split(";")[15]) < 9000.0:

            div += 1
            sm += abs(float(li[i][0].split(";")[3]) - float(li[i][0].split(";")[9]))

    print(sm / div)


