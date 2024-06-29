import torch
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from data.dataset import FraudDataset
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.utils.stop_condition import Threshold, StopOnPlateau
from deepproblog.evaluate import Dataset, Optional, ConfusionMatrix
import torch
import torch.nn as nn
import os


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
        out = nn.Softmax(1)(x)
        return out


def main(train_files, test_files, model):

    train_dataset = FraudDataset(
        "Train", columns=None, target="is_fraud", files=train_files
    )
    test_dataset = FraudDataset(
        "Test", columns=None, target="is_fraud", files=test_files
    )
    # Define the architecture of the neural network
    input_size = ...  # Specify the input size according to your data
    hidden_size = ...  # Specify the size of the hidden layer(s)
    output_size = 1  # Output size (fraud or not fraud)
    lr = 0.001
    batch_size = 64

    loader = DataLoader(train_dataset, batch_size)
    # Create the MLP neural network
    network = DNN(94, 64, 32, 16, 2)

    fraud_net = Network(network, "fraud_net", batching=True)
    fraud_net.optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # Initialize the DeepProbLog model
    path_to_model = f"{os.path.abspath(os.path.dirname(os.path.dirname(__file__)))}/dpl_models/{model}"
    model = Model(path_to_model, [fraud_net])
    model.add_tensor_source("Train", train_dataset)
    model.add_tensor_source("Test", test_dataset)
    model.set_engine(ExactEngine(model), cache=True)

    # Load datasets
    train_obj = train_model(
        model,
        loader,
        StopOnPlateau("Accuracy", warm_up=2, patience=20)
        | Threshold("Accuracy", 1.0, duration=2),
        log_iter=len(train_dataset) // batch_size,
        test_iter=2 * len(train_dataset) // batch_size,
        test=lambda x: [
            ("Accuracy", get_confusion_matrix(x, test_dataset, verbose=1).accuracy())
        ],
        infoloss=0.25,
    )


if __name__ == "__main__":

    path_to_data = f"{os.path.abspath(os.path.dirname(os.path.dirname(__file__)))}/data/downsampled"
    train_files = [
        f"{path_to_data}/fraudTrainPreprocessedDownsampled.csv",
        f"{path_to_data}/fraudTrainPreprocessedScaledDownsampled.csv",
    ]
    test_files = [
        f"{path_to_data}/fraudTestPreprocessedOriginal.csv",
        f"{path_to_data}/fraudTestPreprocessedScaledOriginal.csv",
    ]

    main(train_files, test_files, "model_experiment2.pl")
