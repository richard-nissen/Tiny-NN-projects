# EDA and Preprocessing of Iris Data

To teach myself some basic EDA I've looked at the [Iris dataset from UCI](https://archive.ics.uci.edu/ml/datasets/iris).

## Process

First I looked at the data info and description using pandas' dataframe object, then I tried indexing into the dataframe and could split it into two dataframes, with sepal length less than or greater than 5 and printed the describtion of those datadrames.

Then I plotted the different classes as a function of their sepal width and length, this plot is shown here: ![sepal widtch vs sepal length by species](figures/boxplot_of_sepal_length_width_and_petal_length_width.png)