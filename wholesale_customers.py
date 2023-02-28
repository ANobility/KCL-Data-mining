# Part 2: Cluster Analysis

import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'wholesale_customers.csv'.

def read_csv_2(data_file):
    df = pd.read_csv(data_file)
    df = df.drop('Channel', axis=1)
    df = df.drop('Region', axis=1)
    return df


# Return a pandas dataframe with summary statistics of the data.
# Namely, 'mean', 'std' (standard deviation), 'min', and 'max' for each attribute.
# These strings index the new dataframe columns. 
# Each row should correspond to an attribute in the original data and be indexed with the attribute name.
def summary_statistics(df):
    means = []
    stds = []
    mins = []
    maxs = []
    data = {}
    for i in df.columns:
        means.append(int(df[i].mean()))
        stds.append(int(df[i].std()))
        mins.append(df[i].min())
        maxs.append(df[i].max())
    data['mean'] = means
    data['std'] = stds
    data['min'] = mins
    data['max'] = maxs
    df1 = pd.DataFrame(data, df.columns)
    return df1


# Given a dataframe df with numeric values, return a dataframe (new copy)
# where each attribute value is subtracted by the mean and then divided by the
# standard deviation for that attribute.

def standardize(df):
    df = df.astype(np.float64)
    new_df = df.copy(deep=True)
    summary = summary_statistics(df)
    n = len(df)
    attributes = df.columns
    for i in range(n):
        for j in range(len(attributes)):
            a = (df.loc[i][j] - summary.loc[attributes[j]]['mean']) / summary.loc[attributes[j]]['std']
            new_df.loc[i][j] = a
    return new_df


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans.
# y should contain values in the set {0,1,...,k-1}.

def kmeans(df, k):
    km = cluster.KMeans(n_clusters=k,
                        n_init=10,
                        max_iter=300,
                        random_state=0)
    km.fit(df)
    y = pd.Series(km.labels_)
    return y


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using kmeans++.
# y should contain values from the set {0,1,...,k-1}.
def kmeans_plus(df, k):
    km = cluster.KMeans(n_clusters=k,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        random_state=0)
    km.fit(df)
    y = pd.Series(km.labels_)
    return y


# Given a dataframe df and a number of clusters k, return a pandas series y
# specifying an assignment of instances to clusters, using agglomerative hierarchical clustering.
# y should contain values from the set {0,1,...,k-1}.
def agglomerative(df, k):
    ac = cluster.AgglomerativeClustering(n_clusters=k,
                                         affinity='euclidean',
                                         linkage='average')
    ac.fit(df)
    y = pd.Series(ac.labels_)
    return y


# Given a data set X and an assignment to clusters y
# return the Solhouette score of the clustering.

def clustering_score(X, y):
    m = metrics.silhouette_score(X, y, metric='euclidean')
    return m


# Perform the cluster evaluation described in the coursework description.
# Given the dataframe df with the data to be clustered,
# return a pandas dataframe with an entry for each clustering algorithm execution.
# Each entry should contain the: 
# 'Algorithm' name: either 'Kmeans' or 'Agglomerative', 
# 'data' type: either 'Original' or 'Standardized',
# 'k': the number of clusters produced,
# 'Silhouette Score': for evaluating the resulting set of clusters.
def cluster_evaluation(df):
    stand_df = standardize(df)
    k = [3, 5, 10]
    map = {}

    for i in range(len(k)):
        map[i] = ['Kmeans', 'Original', k[i], clustering_score(df, kmeans(df, k[i]))]
        map[i + 3] = ['Kmeans', 'Standardized', k[i], clustering_score(stand_df, kmeans(stand_df, k[i]))]
        map[i + 6] = ['Agglomerative', 'Original', k[i], clustering_score(df, agglomerative(df, k[i]))]
        map[i + 9] = ['Agglomerative', 'Standardized', k[i], clustering_score(stand_df, agglomerative(stand_df, k[i]))]
    new_df = pd.DataFrame(map.values(), columns=['Algorithm', 'data', 'k', 'Silhouette Score'])
    return new_df


# Given the performance evaluation dataframe produced by the cluster_evaluation function,
# return the best computed Silhouette score.
def best_clustering_score(rdf):
    best = rdf['Silhouette Score'].max()
    return best


# Run some clustering algorithm of your choice with k=3 and generate a scatter plot for each pair of attributes.
# Data points in different clusters should appear with different colors.


def scatter_plots(df):
    stand_df = standardize(df)
    y = agglomerative(stand_df, 3)
    for i in range(len(y)):
        if y[i] == 0:
            y[i] = 'r'
        elif y[i] == 1:
            y[i] = 'g'
        elif y[i] == 2:
            y[i] = 'b'
    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            df.plot.scatter(x=df.columns[i], y=df.columns[j], c=y)
            f = plt.gcf()
            f.savefig('{}.jpg'.format(df.columns[i] + df.columns[j]))
            f.clear()
