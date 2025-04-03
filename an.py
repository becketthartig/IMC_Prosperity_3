list1 = []
list2 = []
list3 = []
list4 = []
list5 = []
list6 = []
list7 = []
list8 = []

with open("moneyyyy.txt", "r") as file:

    # sm = 0
    # div = 0
    # mx = float('-inf')

    signal_tracker = 0

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

            list8 = [20 if (i >= 35 and i < 55) or
                           (i >= 106 and i < 168) or
                           (i >= 386 and i < 391) or
                           (i >= 411 and i < 426) or
                           (i >= 450 and i < 465) or
                           (i >= 601 and i < 631) or
                           (i >= 764 and i < 824) or
                           (i >= 877 and i < 934) or
                           (i >= 960 and i < 965) or
                           (i >= 1227 and i < 1257) or 
                           (i >= 1323 and i < 1539) or
                           (i >= 1556 and i < 1838) or
                           (i >= 1930 and i < 1967) else -20 for i in range(2000)]
            
            for i in range(2000):
                if (i >= 134 and i < 168) or (i >= 217 and i < 344) or (i >= 661 and i < 748) or (i >= 783 and i < 824) or (i >= 983 and i < 1173):
                    list8[i] = 0

            # if roc >= 0.011:
            #     signal_tracker = signal_tracker + 1 if signal_tracker >= 0 else 1 
            #     if signal_tracker > 2:
            #         list8.append(20)
            #     else:
            #         list8.append(0)
            # elif roc <= -0.011:
            #     signal_tracker = signal_tracker - 1 if signal_tracker <= 0 else -1 
            #     if signal_tracker < -2:
            #         list8.append(-20)
            #     else:
            #         list8.append(0)
            # elif s >= 0.05 and roc >= 0:
            #     signal_tracker = signal_tracker + 1 if signal_tracker >= 0 else 1 
            #     if signal_tracker > 2:
            #         list8.append(20)
            #     else:
            #         list8.append(0)
            # elif s <= -0.05 and roc <= 0:
            #     signal_tracker = signal_tracker - 1 if signal_tracker <= 0 else -1 
            #     if signal_tracker < -2:
            #         list8.append(-20)
            #     else:
            #         list8.append(0)
            # else:
            #     signal_tracker = 0
            #     list8.append(0)

            # print(signal_tracker)

            # if roc > 

            

    # print(sm / div)
    # print(mx)


import matplotlib.pyplot as plt

def plot_three_lists(list1, list2, list3):
    plt.figure(figsize=(10, 5))


    plt.plot(list6, label="limit", color='black')
    plt.plot(list5, label="limit", color='black')
    plt.plot(list4, label="limit", color='black')
    # plt.plot(list8, label="test", color='cyan')
    plt.plot(list1, label="linear regression", color='b')
    plt.plot(list2, label="mid price", color='g')
    plt.plot(list3, label="position", color='r')
    plt.plot(list7, label="derivative", color='orange')

    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.title("Trends of Three Lists")
    plt.legend()
    plt.show()


def plot_colored_points(values, binary_labels, y_values):
    plt.figure(figsize=(10, 6))

    for val, y, label in zip(values, y_values, binary_labels):
        if label == 20:
            color = 'red'
        elif label == 0:
            color = 'green'
        else:
            color = 'blue'
        plt.scatter(val, y, color=color, s=50)

    plt.axhline(0, color='black', linewidth=0.5)  # Reference line
    plt.xlim(-5, 5)  # Set x-axis limits based on value range
    plt.ylim(-5, 5)  # Set y-axis limits based on y_values range
    plt.xlabel("Values")
    plt.ylabel("Y Values")
    plt.title("Number Line with Y Values from Third List")
    plt.show()

# plot_colored_points(list1, list8, list7)

# print(max(list1) / 20)
# print(sorted([l / 20 if l / 20 > 0.1 else 0 for l in list1]))

plot_three_lists(list1, list2, list3)



