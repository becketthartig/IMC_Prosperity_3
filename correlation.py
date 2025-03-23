import csv

with open('c64baf0e-bf54-4a56-b807-51139d48fab0.csv', 'r') as file:
    csvreader = csv.reader(file)
    li = list(csvreader)

    nex = {}

    for i in range(1, len(li) - 1):
        if float(li[i][0].split(";")[15]) == 10001.0:

            ii = i+1 
            n = 0
            while n < 9000:
                n = float(li[ii][0].split(";")[15])
                ii += 1
            if n in nex.keys():
                nex[n] += 1
            else:
                nex[n] = 1

    print(nex)
    print(sum(nex.values()))


