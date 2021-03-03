'''
Simple Model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import os 
path = os.getcwd() 
parent = os.path.dirname(path) 
os.chdir(parent)


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
    
def train(net, x, y, epochs=10):
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        
        xnp = x.values
        ynp = y.values
        x_t = torch.from_numpy(xnp)
        y_t = torch.from_numpy(ynp)
       
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(x_t.float())
        loss = criterion(outputs, y_t.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
            
def test(net, x, y):
    """Validate the network on the entire test set."""
    criterion = nn.BCELoss()
    correct = 0
    total = 0
    loss = 0.0
   
    with torch.no_grad():
        
        xnp = x.values
        ynp = y.values
        x_t = torch.from_numpy(xnp)
        y_t = torch.from_numpy(ynp)
        
        outputs = net(x_t.float())
        loss += criterion(outputs, y_t.float()).item()
        predicted = outputs.data
        predicted_score = np.zeros(y_t.size(0))
        for i in range(predicted.size(0)):
            predicted_score[i] = predicted [i]
        total = y_t.size(0)
        correct = predicted_score.sum().item()
    accuracy = correct / total
    return loss, accuracy


def load_data():
    data = pd.read_csv('data/patient.csv')
    y = data['hospitaldischargestatus']
    X = data.drop(columns=['hospitaldischargestatus'])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train = X_train.drop(columns=['Unnamed: 0'])
    X_test = X_test.drop(columns=['Unnamed: 0'])

    trainset = pd.concat([X_train, y_train], axis=1)
    testset = pd.concat([X_test, y_test], axis=1)

    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=16)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=16)
    
    return X_train, X_test, y_train, y_test
    
            
    


if __name__ == "__main__":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    X_train, X_test, y_train, y_test = load_data()
    #X_train = X_train.drop(columns=['Unnamed: 0'])
    #X_test = X_test.drop(columns=['Unnamed: 0'])
    print("Start training")
    train(net=Net(), x=X_train, y=y_train, epochs=10)
    print("Evaluate model")
    loss, accuracy = test(net=Net(), x=X_test, y=y_test)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)                   
    
    
