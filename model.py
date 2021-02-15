
from data_analysis import data_2_inputs,data_2_labels


import sklearn
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

#Using data_2 as an example here

X_train, x_test, Y_train, y_test = train_test_split(data_2_inputs,
                                                    data_2_labels,
                                                    test_size=0.2,
                                                    random_state=0)


#print(Y_train.head())



Xtrain_torch = torch.from_numpy(X_train.values).float()
Xtest_torch = torch.from_numpy(x_test.values).float()


Ytrain_torch = torch.from_numpy(Y_train.values).view(1,-1)[0]
Ytest_torch = torch.from_numpy(y_test.values).view(1,-1)[0]


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size):
        super(Net, self).__init__()
        #Model based on 2 fully connected linear layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size) 
    
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x)) 
        
        x = torch.sigmoid(self.fc2(x)) 
        
        #Pass the output to log Softmax function
        return F.log_softmax(x, dim=-1)

model = Net(input_size = 32,hidden_size = 10,output_size = 2)


optimizer = torch.optim.Adam(model.parameters(),lr=0.003)

loss_fn = nn.NLLLoss()




epoch_data = []
epochs = 1001




def fit(x,y): 

    for epoch in range(1, 1001):

        #Zero out gradients for every epoch
        #Forward pass
        optimizer.zero_grad()

        Ypred = model(x)

        #Calculate the loss on prediction and back propagate to calculate gradients
        loss = loss_fn(Ypred , y)
        loss.backward()
        
        #update model parameters
        optimizer.step()

        epoch_data.append([epoch, loss.data.item()])
        
        #print for every 100 epochs
        if epoch % 100 == 0:
            print ('epoch - %d  train loss - %.2f'  % (epoch, loss.data.item()))


fit(Xtrain_torch,Ytrain_torch)


def evaluate(x,y): 

    with torch.no_grad():
            model.eval()
            y_pred = model(x)
    
    #Prediction is based on the highest probability
    _,pred = y_pred.data.max(1)


    #Compare prediction against actual labels
    accuracy = pred.eq(y.data).sum().item() / y_test.values.size

    return accuracy


print(evaluate(Xtest_torch,Ytest_torch))




import matplotlib.pyplot as plt


df_epochs_data = pd.DataFrame(epoch_data, 
                              columns=["epoch", "train_loss"])

x = df_epochs_data["epoch"]
y=df_epochs_data["train_loss"]
#Graph1 train loss over epoch
plt.plot(x,y)
plt.show()



