import numpy as np
import torch
from collections import OrderedDict

import simple_model
import flwr as fl

class SimpleClient(fl.client.NumPyClient):
    

    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

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
        simple_model.train(self.model, self.X_train, self.y_train, epochs=10)
        return self.get_parameters(), len(self.y_train)

    def evaluate(self, parameters, config):
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = simple_model.test(self.model, self.X_test, self.y_test)
        return len(self.y_test), float(loss), float(accuracy)
    
def main() -> None:
    """Load data, start SimpleClient."""

    # Load model and data
    model = simple_model.Net()
    X_train, X_test, y_train, y_test = simple_model.load_data()

    # Start client
    client = SimpleClient(model, X_train, X_test, y_train, y_test)
    fl.client.start_numpy_client("0.0.0.0:8080", client)


if __name__ == "__main__":
    main()
