# Gini impurity calculation
def gini(y):
    if len(y) == 0:
        return 0
    p = sum(y) / len(y)
    return 2 * p * (1 - p)

# Finds the best feature and threshold to split on
def best_split(X, y):
    print("begin")
    best_feat = None
    best_thresh = None
    best_gini = float('inf')

    n_samples = len(X)
    n_features = len(X[0])

    for feature in range(n_features):
        print("began feature")
        # Get all values of the feature
        values = [row[feature] for row in X]
        sorted_pairs = sorted(zip(values, y))

        for i in range(1, n_samples):
            if sorted_pairs[i][0] == sorted_pairs[i - 1][0]:
                continue
            threshold = (sorted_pairs[i][0] + sorted_pairs[i - 1][0]) / 2
            left_y = [label for val, label in sorted_pairs if val <= threshold]
            right_y = [label for val, label in sorted_pairs if val > threshold]
            weighted_gini = len(left_y)/n_samples * gini(left_y) + len(right_y)/n_samples * gini(right_y)

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feat = feature
                best_thresh = threshold

    print("ret")

    return best_feat, best_thresh

# Builds the decision tree recursively
def build_tree(X, y, depth=0, max_depth=10, min_samples=1):
    # Stopping conditions
    if len(set(y)) == 1:
        return {"leaf": True, "class": y[0]}
    if len(X) <= min_samples or depth == max_depth:
        majority = 1 if sum(y) >= len(y) / 2 else 0
        return {"leaf": True, "class": majority}

    feat, thresh = best_split(X, y)
    if feat is None:
        majority = 1 if sum(y) >= len(y) / 2 else 0
        return {"leaf": True, "class": majority}

    # Split data
    left_X = [row for row in X if row[feat] <= thresh]
    left_y = [y[i] for i in range(len(X)) if X[i][feat] <= thresh]
    right_X = [row for row in X if row[feat] > thresh]
    right_y = [y[i] for i in range(len(X)) if X[i][feat] > thresh]


    # Build subtrees
    return {
        "leaf": False,
        "feature": feat,
        "threshold": thresh,
        "left": build_tree(left_X, left_y, depth + 1, max_depth, min_samples),
        "right": build_tree(right_X, right_y, depth + 1, max_depth, min_samples)
    }

# Prediction function using the built tree
def predict_one(tree, x):
    while not tree["leaf"]:
        if x[tree["feature"]] <= tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree["class"]

# Batch prediction
def predict(tree, X):
    return [predict_one(tree, x) for x in X]


import csv

hist = []
def read_in(p):
    with open(p, 'r') as file:

        csvreader = csv.reader(file)
        li = list(csvreader)

        for i in range(1, len(li)):

            line = li[i][0].split(";")
            product = line[2]

            if product == "MAGNIFICENT_MACARONS":
                mp = float(line[15])
                hist.append(mp)



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
            sug = float(line[6])
            sun = float(line[7])

            implied_bids.append(bid - export - trans)
            implied_asks.append(ask + imp + trans)
            sugs.append(sug)
            suns.append(sun)

read_in("data/round4/prices_round_4_day_1.csv")
read_in("data/round4/prices_round_4_day_2.csv")
read_in("data/round4/prices_round_4_day_3.csv")

read_in_implied("data/round4/observations_round_4_day_1.csv")
read_in_implied("data/round4/observations_round_4_day_2.csv")
read_in_implied("data/round4/observations_round_4_day_3.csv")

ROC_suns = [0]
for i in range(1, len(suns)):
    ROC_suns.append(suns[i] - suns[i - 1])

featurezzz = []
ranges = [(0, 760),
          (1740, 4010),
          (5830, 6380),
          (6120, 7280),
          (10450, 12370), 
          (16160, 18430), 
          (20400, 23360),
          (25760, 29030)]

marks = []
for i in range(len(hist)):
    featurezzz.append([hist[i], suns[i], ROC_suns[i], sugs[i]])
    flag = True
    for r in ranges:
        if i >= r[0] and i <= r[1]:
            marks.append(1)
            flag = False
            break
    if flag:
        marks.append(0)

import pprint

pprint.pprint({'leaf': False, 'feature': 0, 'threshold': 615.75, 'left': {'leaf': False, 'feature': 0, 'threshold': 605.75, 'left': {'leaf': False, 'feature': 0, 'threshold': 599.75, 'left': {'leaf': True, 'class': 1}, 'right': {'leaf': True, 'class': 1}}, 'right': {'leaf': False, 'feature': 3, 'threshold': 198.34524549289813, 'left': {'leaf': True, 'class': 0}, 'right': {'leaf': True, 'class': 1}}}, 'right': {'leaf': False, 'feature': 1, 'threshold': 44.995000000000005, 'left': {'leaf': False, 'feature': 2, 'threshold': 0.019999999999999574, 'left': {'leaf': True, 'class': 1}, 'right': {'leaf': True, 'class': 0}}, 'right': {'leaf': False, 'feature': 2, 'threshold': -0.015000000000000568, 'left': {'leaf': True, 'class': 1}, 'right': {'leaf': True, 'class': 0}}}})

# tree = build_tree(featurezzz, marks, max_depth=3)
# print(tree)

tree = {'leaf': False, 'feature': 0, 'threshold': 615.75, 'left': {'leaf': False, 'feature': 0, 'threshold': 605.75, 'left': {'leaf': False, 'feature': 0, 'threshold': 599.75, 'left': {'leaf': True, 'class': 1}, 'right': {'leaf': True, 'class': 1}}, 'right': {'leaf': False, 'feature': 3, 'threshold': 198.34524549289813, 'left': {'leaf': True, 'class': 0}, 'right': {'leaf': True, 'class': 1}}}, 'right': {'leaf': False, 'feature': 1, 'threshold': 44.995000000000005, 'left': {'leaf': False, 'feature': 2, 'threshold': 0.019999999999999574, 'left': {'leaf': True, 'class': 1}, 'right': {'leaf': True, 'class': 0}}, 'right': {'leaf': False, 'feature': 2, 'threshold': -0.015000000000000568, 'left': {'leaf': True, 'class': 1}, 'right': {'leaf': True, 'class': 0}}}}

wrong = 0
for i, m in enumerate(predict(tree, featurezzz)):
    if marks[i] != m:
        wrong += 1

print(wrong)

import bruh

bruh.plot_time_series2(predict(tree, featurezzz), marks)