import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/diabetes.csv')

# Looking at the first 5 rows of the data
print(df.head())

# I can see that there are 8 features of the set and 1 outcome column that says whether the patient has diabetes or not.

# I want to make a histogram of the different features to see how they are distributed. I will use the pandas hist() function to do this.

df.hist(figsize=(15,15))
plt.show()

