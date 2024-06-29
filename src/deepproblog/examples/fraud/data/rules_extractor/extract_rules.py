from sklearn.tree import _tree
import pickle
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle5 as pickle

df_train = pd.read_csv("../archive/fraudTrain.csv")
df_train = df_train.sample(frac=1).reset_index(drop=True)
df_train = df_train.loc[:, ~df_train.columns.str.contains("^Unnamed")]
df_train.drop(
    columns=[
        "merchant",
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "job",
        "trans_num",
        "unix_time",
    ],
    inplace=True,
)

# Set max depth for decision trees
max_depth = 3


def preprocess(df):

    # Date transformations
    date_column = pd.to_datetime(df["trans_date_trans_time"])
    df.drop(columns=["trans_date_trans_time"], inplace=True)
    df["hour"] = date_column.dt.hour
    df["day"] = date_column.dt.day
    df["month"] = date_column.dt.month
    # df['year'] = date_column.dt.year # Do not include year

    # Age transformations
    birth_column = pd.to_datetime(df["dob"])
    df.drop(columns=["dob"], inplace=True)
    age = date_column.dt.year - birth_column.dt.year
    df["age"] = age
    df["age"] = df["age"].apply(lambda x: f"{(x//5)*5}-{(x//5)*5 + 5}")

    df = pd.get_dummies(df, columns=["category", "gender", "state", "age"], dtype=int)

    return df


def split_dataset(df):

    X = df[df.columns.difference(["is_fraud"])]
    y = df.is_fraud

    return X, y


def train_model(X, y):

    # Fit the classifier with max_depth=3
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model = clf.fit(X, y)

    return model


def tree_to_code(tree, feature_names):
    """
    Outputs a decision tree model as a Python function

    Parameters:
    -----------
    tree: decision tree model
        The decision tree to represent as a function
    feature_names: list
        The feature names of the dataset used for building the decision tree
    """

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []

    for path in paths:
        rule = "if "

        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: " + str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


if __name__ == "__main__":

    df = preprocess(df_train)
    X, y = split_dataset(df)
    model = train_model(X, y)

    features = model.feature_names_in_
    classes = model.classes_
    # Creating the tree plot
    tree.plot_tree(model, filled=True)
    plt.rcParams["figure.figsize"] = [10, 10]

    rules = get_rules(model, features, classes)
    for r in rules:
        print(r)
