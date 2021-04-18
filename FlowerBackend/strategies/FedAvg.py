
from typing import List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

import torch
from .FedStrategy import FedStrategy
import numpy as np
DEFAULT_SERVER_ADDRESS = "[::]:8080"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "data/patient.csv"

class FedAvg(FedStrategy):

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
        initial_parameters = None,
        mode = 'datasize',
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            )
        self.mode = mode

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None

        if not self.accept_failures and failures:
            return None
        if self.mode == 'datasize':

            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
                for client, fit_res in results
            ]
        if self.mode == 'accuracy':

            weights_results = [
                (parameters_to_weights(fit_res.parameters), fit_res.fit_duration)
                for client, fit_res in results
            ]

        weights = aggregate(weights_results)
        if weights is not None:
            print(f"Saving round {rnd} weights...")
            np.savez(f"round-{rnd}-weights.npz", *weights)
        return weights
