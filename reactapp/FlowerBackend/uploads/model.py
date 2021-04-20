# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


# mypy: ignore-errors
# pylint: disable=W0223



from typing import Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import flwr as fl


# pylint: disable=unsubscriptable-object,bad-option-value,R1725
class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        # Model based on 2 fully connected linear layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))

        x = torch.sigmoid(self.fc2(x))

        # Pass the output to log Softmax function
        return F.log_softmax(x, dim=-1)

class MyDataset(Dataset):

    def __init__(self, train_data, labels):
        self.x_train = train_data
        self.y_train = labels

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, item):
        return self.x_train[item], self.y_train[item]

class Loader():

    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(self.path)
        self.feature = self.data.drop(columns=['hospitaldischargestatus'])
        self.input_size = self.feature.shape[1]

    def load_model(self) -> Net:
        """Load a simple CNN."""
        return Net(input_size = self.input_size,hidden_size = 10,output_size = 2)

# pylint: disable=unused-argument
    def load_data(self) :
        """Load CIFAR-10 (training and test set)."""
        feature = np.array(self.feature, dtype=np.float32)
        labels = np.array(self.data['hospitaldischargestatus'], dtype=np.float32)

        feature_ = torch.tensor(feature)
        labels_ = torch.tensor(labels).view(-1, 1)


        train_and_dev = MyDataset(feature_, labels_ )

        train_examples = round(len(train_and_dev) * 0.8)
        dev_examples = len(train_and_dev) - train_examples

        train_dataset, dev_dataset = random_split(train_and_dev,
                                              (train_examples,
                                               dev_examples))
        return train_dataset, dev_dataset


def train(
    net: Net,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
) -> None:
    """Train the network."""
    # Define loss and optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003)

    loss_fn = nn.NLLLoss()

    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        for i, (features, target) in enumerate(trainloader):

            feature = features.to(device)
            target = target.squeeze(1)
            target = target.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(feature).squeeze(1)
            loss = loss_fn(outputs, target.type(torch.LongTensor))
            loss.backward()
            optimizer.step()

            # print statistics
            #if epoch % 100 == 0:
            print('epoch - %d  train loss - %.2f' % (epoch, loss.data.item()))


def test(
    net: Net,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    loss_fn = nn.NLLLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            features, labels = data[0].to(device), data[1].to(device)
            labels = labels.squeeze(1)
            outputs = net(features).squeeze(1)
            loss += loss_fn(outputs, labels.type(torch.LongTensor)).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
