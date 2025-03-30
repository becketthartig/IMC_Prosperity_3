list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []

with open("/Users/becketthartig/dev/P3 - Literal Zero/nomoney10.txt", "r") as file:

    # sm = 0
    # div = 0
    # mx = float('-inf')

    for l in file:
        if l[2:10] == '"lambdaL':
            vals = l.split('::')
            roc = float(vals[-5])
            s = float(vals[-2])
            list7.append(roc * 80)
            list1.append(s * 20)
            list2.append((float(vals[-3]) - 2010) * 2)
            list3.append(float(vals[-4]) / 2)
            list4.append(0)
            list5.append(0.055 * 20)
            list6.append(-0.055 * 20)

            if roc >= 0.0125 and s >= 0:
                list8.append(20)
            elif roc <= -0.0125  and s <= 0:
                list8.append(-20)
            elif s >= 0.055:
                list8.append(20)
            elif s <= -0.055:
                list8.append(-20)
            else:
                list8.append(0)

            # if roc > 

            

    # print(sm / div)
    # print(mx)


import matplotlib.pyplot as plt

def plot_three_lists(list1, list2, list3):
    plt.figure(figsize=(10, 5))


    plt.plot(list6, label="List 3", color='black')
    plt.plot(list5, label="List 3", color='black')
    plt.plot(list4, label="List 3", color='black')
    plt.plot(list8, label="buy/sell", color='cyan')
    plt.plot(list1, label="List 1", color='b')
    plt.plot(list2, label="List 2", color='g')
    plt.plot(list3, label="List 3", color='r')
    plt.plot(list7, label="List 7", color='orange')

    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.title("Trends of Three Lists")
    plt.legend()
    plt.show()


plot_three_lists(list1, list2, list3)



