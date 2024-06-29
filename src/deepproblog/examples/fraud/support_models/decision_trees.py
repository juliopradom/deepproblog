import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle5 as pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import os


def train_model(X, y):

    # Fit the classifier with max_depth=3
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    model = clf.fit(X, y)

    return model


def main(training_dataset_path, test_dataset_path):

    df_train = pd.read_csv(training_dataset_path)
    df_test = pd.read_csv(test_dataset_path)
    df_train = df_train.loc[:, ~df_train.columns.str.contains("^Unnamed")]
    df_train = df_train.loc[:, ~df_train.columns.str.contains("^index")]
    df_test = df_test.loc[:, ~df_test.columns.str.contains("^Unnamed")]
    df_test = df_test.loc[:, ~df_test.columns.str.contains("^index")]

    features = list(df_train.columns)
    features.remove("is_fraud")
    X_train, y_train = df_train[features], df_train["is_fraud"]
    features = list(df_test.columns)
    features.remove("is_fraud")
    X_val, y_val = df_test[features], df_test["is_fraud"]

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)

    # Print confusion matrix
    print("Confusion Matrix:")
    print(cm)
    print(accuracy_score(y_val, y_pred))


if __name__ == "__main__":

    path_to_data = f"{os.path.abspath(os.path.dirname(os.path.dirname(__file__)))}/data/downsampled"
    main(
        f"{path_to_data}/fraudTrainPreprocessedDownsampled.csv",
        f"{path_to_data}/fraudTestPreprocessedOriginal.csv",
    )
