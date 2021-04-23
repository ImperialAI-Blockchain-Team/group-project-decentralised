# Please Include an empty line at the start and at the end of your file

import torch
import torch.nn as nn
import pandas as pd
import flower as fl
from typing import Tuple


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pass

    def forward(self, x):
        pass

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        pass

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        pass

class Loader():
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(self.path)

    def load_model(self) -> Net:
        """Returns an instance of Net."""
        pass

    def load_data(self) -> Tuple:
        """Load data.
            Returns: train_dataset, test_dataset
            train_dataset and test_dataset must be supported by torch.utils.data.DataLoader.
            WARNING: testing will be conducted on the server side, test_dataset can have 0 samples.
        """
        pass



def train(
    model: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device) -> None:
    """Train the network."""
    pass


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device) -> Tuple[float, float]:
    """Validate the network on the entire test set.
        Returns: loss and accuracy.
    """
    pass