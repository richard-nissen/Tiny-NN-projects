import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(URL, names=['sepal length','sepal width','petal length','petal width','target'])

# Find missing values
print("Missing values : \n{}".format(df.isnull().any()))

# Delete the sepal length feature from 10 random rows
df.loc[np.random.choice(df.index, 10), 'sepal length'] = None

# Find missing values
print("Missing values : \n{}".format(df.isnull().any()))

# Drop rows with missing values
print("Iris data set info before dropping rows : \n{}".format(df.shape[0]))
df2 = df.dropna()
print("Iris data set info after dropping rows: \n{}".format(df2.shape[0]))

# Fill missing values with the mean value
df3 = df.fillna(df.mean())
print("Iris data set info after filling missing values with the mean value: \n{}".format(df3.shape[0]))
print("Check if there are still missing values : \n{}".format(df3.isnull().any()))
