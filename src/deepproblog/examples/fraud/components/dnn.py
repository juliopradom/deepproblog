import torch.nn as nn


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
