import pandas as pd
import matplotlib.pyplot as plt

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(URL, names=['sepal length','sepal width','petal length','petal width','target'])

print("Iris data set info : \n{}".format(df.info()))
print("Iris data set description : \n{}".format(df.describe()))
print("Iris data set head : \n{}".format(df.head(10)))

# Select rows where sepal_length > 5.0
df_big_sepal_length = df.loc[df['sepal length'] > 5.0]
df_small_sepal_length = df.loc[df['sepal length'] <= 5.0]
print("Iris data set where sepal_length > 5.0 : \n{}".format(df_big_sepal_length.describe()))
print("Iris data set where sepal_length <= 5.0 : \n{}".format(df_small_sepal_length.describe()))

markers = ['o', 'x', '^']

ax = plt.axes()

for i, species in enumerate(df['target'].unique()):
    species_data = df[df['target'] == species]
    species_data.plot.scatter(x='sepal length',
                              y='sepal width',
                              marker = markers[i],
                              s = 100,
                              title = 'Sepal Width vs Length by Species',
                              label = species, figsize=(10,7), ax=ax)

plt.savefig('figures/sepal_width_vs_length_by_species.png')
plt.show()


df['petal length'].plot.hist(title = 'Petal Length Histogram')

plt.savefig('figures/petal_length_histogram.png')
plt.show()

df.plot.box(title = 'Boxplot of Sepal Length & Width, and Petal Length & Width')

plt.savefig('figures/boxplot_of_sepal_length_width_and_petal_length_width.png')
plt.show()