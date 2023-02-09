import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/diabetes.csv')

"""
# Initial EDA
print(df.head())
"""

"""
# Overview of the data set
df.hist(figsize=(10,10))
plt.show()
"""

"""
# Plotting the relationship between each feature and the outcome

# Create a subplot of 3 rows and 3 columns
plt.subplots(3, 3, figsize=(10,10))

# Goes through each feature and plots the distribution of the feature for each outcome
for i, feature in enumerate(list(df.columns)[:-1]):
    # Create a new subplot for each iteration
    ax = plt.subplot(3, 3, i+1)
    # Remove the axis labels
    ax.yaxis.set_ticklabels([])
    # Plot the distribution of the feature for each outcome, where hist=False means that we are not plotting a histogram, but a density plot
    # axlabel=False means that we are not plotting the x-axis label (since we are plotting multiple plots in the same subplot)
    # kde_kws is where we can specify the style of the density plot
    sns.distplot(df.loc[df.Outcome == 0][feature], hist=False, axlabel=False, kde_kws={'linestyle':'-', 'color':'black', 'label':'No Diabetes'})
    sns.distplot(df.loc[df.Outcome == 1][feature], hist=False, axlabel=False, kde_kws={'linestyle':'--', 'color':'black', 'label':'Diabetes'})
    # Set the title of the subplot
    ax.set_title(feature)

# Hide the 9th subplot (bottom right) since there are only 8 plots
plt.subplot(3, 3, 9).set_visible(False)

plt.show()
"""

# Data Preprocessing

# First I want to check if there are any missing values
print(df.isnull().any())

# It says that there are no missing values, but in our EDA we saw that there were some 0 values in the Glucose, BloodPressure, SkinThickness, Insulin, and BMI columns so let's look at the statistical summary of the data set

print(df.describe())