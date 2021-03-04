from collections import OrderedDict
from typing import Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
import flwr as fl
import matplotlib.pyplot as plt



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

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

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
    
    def load_test_data(self):
        feature = np.array(self.feature, dtype=np.float32)
        labels = np.array(self.data['hospitaldischargestatus'], dtype=np.float32)

        feature_ = torch.tensor(feature)
        labels_ = torch.tensor(labels).view(-1, 1)
        
        testset = MyDataset(feature_, labels_ )
        
        return testset


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
            if epoch % 100 == 0:
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

def Main(
    epochs: int,
    batch_size: int,
    data_path: str
) -> np.array:
    
    SEED = 1

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  
    Total_Model = Loader(data_path)
    net = Total_Model.load_model()
    trainset, valset = Total_Model.load_data()
    
    BATCH_SIZE = batch_size
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False)
    
    
    train(net, trainloader, epochs, device)
    val_loss, val_accuracy = test(net, valloader, device)
        
        
    return net.get_weights()    
        
    #fig, ax = plt.subplots()
    #ax.plot(range(EPOCHS),epoch_train_loss,label="Train")
    #ax.plot(range(EPOCHS),epoch_val_loss,label="Validation")  
    
    #plt.legend(fontsize = 8)
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')  

