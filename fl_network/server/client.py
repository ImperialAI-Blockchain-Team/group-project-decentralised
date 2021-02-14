import numpy as np
import torch
from collections import OrderedDict

import simple_model
import flwr as fl

class CifarClient(fl.client.NumPyClient):
    

    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self):
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        simple_model.train(self.model, self.trainloader, epochs=1)
        return self.get_parameters(), len(self.trainloader)

    def evaluate(self, parameters, config):
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = simple_model.test(self.model, self.testloader)
        return len(self.testloader), float(loss), float(accuracy)
    
def main() -> None:
    """Load data, start CifarClient."""

    # Load model and data
    model = simple_model.Net()
    trainloader, testloader = simple_model.load_data()

    # Start client
    client = CifarClient(model, trainloader, testloader)
    fl.client.start_numpy_client("0.0.0.0:8080", client)


if __name__ == "__main__":
    main()
