'''
Simple Model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.model_selection import train_test_split
import pandas as pd

import os 
path = os.getcwd() 
parent = os.path.dirname(path) 
os.chdir(parent)

data = pd.read_csv('data/patient.csv')
y = data['hospitaldischargestatus']
X = data.drop(columns=['hospitaldischargestatus'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

trainset = pd.concat([X_train, y_train], axis=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()
    
def train(net, trainloader, epochs=10):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            x, y = data[0], data[1]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = nn.BCELoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            x, y = data[0], data[1]
            outputs = net(x)
            loss += criterion(outputs, y).item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = correct / total
    return loss, accuracy
            
                
    
    
