import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/diabetes.csv')


print(df.head())

df.hist(figsize=(10,10))
plt.show()

