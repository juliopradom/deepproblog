from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import os


def main(training_dataset_path, test_dataset_path):
    # load data
    train_dataset = pd.read_csv(training_dataset_path, delimiter=",")
    test_dataset = pd.read_csv(test_dataset_path, delimiter=",")

    train_dataset = train_dataset.loc[
        :, ~train_dataset.columns.str.contains("^Unnamed")
    ]
    train_dataset = train_dataset.loc[:, ~train_dataset.columns.str.contains("^index")]
    train_dataset = train_dataset.loc[:, ~train_dataset.columns.str.contains("^noisy")]

    test_dataset = test_dataset.loc[:, ~test_dataset.columns.str.contains("^Unnamed")]
    test_dataset = test_dataset.loc[:, ~test_dataset.columns.str.contains("^index")]
    test_dataset = test_dataset.loc[:, ~test_dataset.columns.str.contains("^noisy")]

    y_train = train_dataset["is_fraud"].astype("category")
    y_test = test_dataset["is_fraud"].astype("category")

    X_train = train_dataset.loc[:, ~train_dataset.columns.str.contains("^is_fraud")]
    X_test = test_dataset.loc[:, ~test_dataset.columns.str.contains("^is_fraud")]

    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


if __name__ == "__main__":

    path_to_data = f"{os.path.abspath(os.path.dirname(os.path.dirname(__file__)))}/data/downsampled"
    main(
        f"{path_to_data}/fraudTrainPreprocessedDownsampled.csv",
        f"{path_to_data}/fraudTestPreprocessedOriginal.csv",
    )
