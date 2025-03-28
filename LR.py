import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.DataFrame(data, columns=['Price'])

# Function to create supervised learning dataset
def create_dataset(series, lookback=3):
    X, y = [], []
    for i in range(len(series) - lookback):
        X.append(series[i:i+lookback])  # Take the past 'lookback' time steps
        y.append(series[i+lookback])    # Target is the next value
    return np.array(X), np.array(y)

lookback = 3  # Number of past time steps used for prediction
X, y = create_dataset(df['Price'].values, lookback=lookback)