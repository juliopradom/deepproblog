import torch
import torch.nn as nn
import pandas as pd
from torch.utils import data
import numpy as np
import tqdm
import copy
from sklearn.metrics import classification_report
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch.optim as optim


class DNN(nn.Module):
    def __init__(self, *sizes, activation=nn.ReLU, batch=True):
        super(DNN, self).__init__()
        layers = []
        self.batch = batch
        self.class_weights = None
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation())
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        if not self.batch:
            x = x.unsqueeze(0)
        x = self.nn(x)
        out = nn.Sigmoid()(x)
        return out.float()


class ColumnarDataset(data.Dataset):
    def __init__(self, df, y):
        self.dfconts = df
        self.conts = torch.tensor(self.dfconts[df.columns].astype(float).values).float()
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return [self.conts[int(idx)], self.y[idx]]


def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 20  # number of epochs to run
    batch_size = 64  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = -np.inf  # init to negative infinity
    best_weights = None

    X_val = torch.tensor(X_val.astype(float).values).float()

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch

                X_batch = X_train[int(start) : int(start) + batch_size]
                y_batch = y_train[int(start) : int(start) + batch_size]
                X_batch = torch.tensor(X_batch.astype(float).values).float()
                y_batch = torch.tensor(y_batch.to_list())
                # forward pass
                y_pred = model(X_batch)
                y_pred = torch.squeeze(y_pred).float()
                y_pred = y_pred.to(torch.float64)
                y_batch = y_batch.to(torch.float64)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(loss=float(loss), acc=float(acc))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == torch.tensor(y_val.to_list())).float().mean()
        acc = float(acc)
        print(classification_report(y_val.to_list(), y_pred.round().detach().numpy()))
        print(f"Epoch acc: {acc}")
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc


def main(training_dataset_path, test_dataset_path):

    network = DNN(94, 64, 32, 16, 1)

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

    acc = model_train(network, X_train, y_train, X_val, y_val)
    print("Accuracy (wide): %.2f" % acc)


if __name__ == "__main__":

    path_to_data = f"{os.path.abspath(os.path.dirname(os.path.dirname(__file__)))}/data/downsampled"
    main(
        f"{path_to_data}/fraudTrainPreprocessedScaledDownsampled.csv",
        f"{path_to_data}/fraudTestPreprocessedScaledOriginal.csv",
    )
