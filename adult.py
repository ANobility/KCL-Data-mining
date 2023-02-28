# Part 1: Decision Trees with Categorical Attributes

import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score


# Return a pandas dataframe containing the data set that needs to be extracted from the data_file.
# data_file will be populated with the string 'adult.csv'.
def read_csv_1(data_file):
    df = pd.read_csv(data_file)
    df = df.drop('fnlwgt', axis=1)
    return df


# Return the number of rows in the pandas dataframe df.
def num_rows(df):
    return df.shape[0]


# Return a list with the column names in the pandas dataframe df.
def column_names(df):
    return list(df.columns)


# Return the number of missing values in the pandas dataframe df.
def missing_values(df):
    return df.isnull().sum().sum()


# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
    a = df.columns[df.isnull().any()]
    return list(a)


# Return the percentage of instances corresponding to persons whose education level is
# Bachelors or Masters, by rounding to the third decimal digit,
# in the pandas dataframe df containing the data set in the adult.csv file.
# For example, if the percentage is 0.21547%, then the function should return 0.216.
def bachelors_masters_percentage(df):
    num = 0
    for i in df['education']:
        if i == 'Bachelors' or i == 'Masters':
            num += 1
    return '%.3f' % ((num / num_rows(df)))


# Return a pandas dataframe (new copy) obtained from the pandas dataframe df
# by removing all instances with at least one missing value.
def data_frame_without_missing_values(df):
    return df.dropna().reset_index(drop=True)


# Return a pandas dataframe (new copy) from the pandas dataframe df
# by converting the df categorical attributes to numeric using one-hot encoding.
# The function should not encode the target attribute, and the function's output
# should not contain the target attribute.
def one_hot_encoding(df):
    df = pd.get_dummies(df, columns=df.columns.drop('class'))
    return df


# Return a pandas series (new copy), from the pandas dataframe df,
# containing only one column with the labels of the df instances
# converted to numeric using label encoding. 
def label_encoding(df):
    class_mapping = {label: idx for idx, label in enumerate(set(df['class']))}
    df['class'] = df['class'].map(class_mapping)
    return df['class']


# Given a training set X_train containing the input attribute values
# and labels y_train for the training instances,
# build a decision tree and use it to predict labels for X_train. 
# Return a pandas series with the predicted values. 
def dt_predict(X_train, y_train):
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y = clf.predict(X_train)
    return pd.Series(y)


# Given a pandas series y_pred with the predicted labels and a pandas series y_true with the true labels,
# compute the error rate of the classifier that produced y_pred. 
def dt_error_rate(y_pred, y_true):
    score = '%.3f' % ((1 - accuracy_score(list(y_pred), list(y_true))))
    return score
