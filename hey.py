nums = []
with open("backtests/quick.txt", "r") as file:
    for line in file:
        l = line.split("ZAZA")
        if len(l) == 3:
            nums.append(float(l[1]))


import bruh

bruh.plot_time_series(nums)
