import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from collections import defaultdict
import torch
from collections import OrderedDict
DEFAULT_SERVER_ADDRESS = "[::]:8080"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "uploads/testset.csv"

class FedStrategy(fl.server.strategy.Strategy):

    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn = None,
        on_fit_config_fn = None,
        on_evaluate_config_fn = None,
        accept_failures = True,
        initial_parameters=None,
        model = None
    ) -> None:
        super().__init__()
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.model = model
        self.contrib = defaultdict(list)

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int):
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int):
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients


    def evaluate(self, weights):
        """Evaluate model weights using an evaluation function (if
        provided)."""
        if self.eval_fn is None:
            return None
        return self.eval_fn(weights)

    def get_weights(self, model) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_weights(self, model, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
        )
        model.load_state_dict(state_dict, strict=True)

    def configure_fit(
        self, rnd: int, weights, client_manager
    ):
        """Configure the next round of training."""
        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, fit_ins) for client in clients]


    def configure_evaluate(
        self, rnd: int, weights: Weights, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.eval_fn is not None:
            return []

        parameters = weights_to_parameters(weights)
        config = {}
        if self.on_evaluate_config_fn is not None:

            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        return [(client, evaluate_ins) for client in clients]


    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        if not self.accept_failures and failures:
            return None
        return weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss, evaluate_res.accuracy)
                for _, evaluate_res in results
            ]
        )

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Weights]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters
