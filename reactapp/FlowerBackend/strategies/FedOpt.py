'''Ref:  https://arxiv.org/pdf/2003.00295.pdf'''

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_proxy import ClientProxy

import torch
from .FedStrategy import FedStrategy

import json
DEFAULT_SERVER_ADDRESS = "[::]:8080"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "uploads/testset.csv"

class FedOpt(FedStrategy):
    def __init__(
        self,
        *,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn = None,
        on_fit_config_fn = None,
        on_evaluate_config_fn = None,
        accept_failures = True,
        mode = 'adagrad',
        beta = 0.99,
        initial_parameters = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        tau: float = 1e-9,
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
        self.current_weights = initial_parameters
        self.beta = beta
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.v_t: Optional[Weights] = None

    def __repr__(self) -> str:
        rep = f"FedOpt(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Weights]:
        if not results:
            return None
        if not self.accept_failures and failures:
            return None
        net = self.model.Loader(DATA_ROOT).load_model()
        testset, _ = self.model.Loader(DATA_ROOT).load_data()
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
        for client, fit_res in results:
            self.set_weights(net, parameters_to_weights(fit_res.parameters))
            net.to(DEVICE)
            loss, acc = self.model.test(net, testloader, device=DEVICE)
            self.contrib[fit_res.metrics['cid']].append(acc)

        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        fedavg_aggregate = aggregate(weights_results)

        if fedavg_aggregate is None:
            return None

        aggregated_updates = [
            subset_weights - self.current_weights[idx]
            for idx, subset_weights in enumerate(fedavg_aggregate)
        ]
        delta_t = aggregated_updates
        if not self.v_t:
            self.v_t = [np.zeros_like(subset_weights) for subset_weights in delta_t]

        if self.mode == 'adagrad':
            self.v_t = [
                self.v_t[idx] + np.multiply(subset_weights, subset_weights)
                for idx, subset_weights in enumerate(delta_t)
            ]
        if self.mode == 'yogi':
            self.v_t = [
                self.v_t[idx] - (1 - self.beta)*np.multiply(subset_weights, subset_weights)*np.sign(self.v_t[idx] - np.multiply(subset_weights, subset_weights))
                for idx, subset_weights in enumerate(delta_t)
            ]
        if self.mode == 'adam':
            self.v_t = [
                self.beta*self.v_t[idx] + (1 - self.beta)*np.multiply(subset_weights, subset_weights)
                for idx, subset_weights in enumerate(delta_t)
            ]

        new_weights = [
            self.current_weights[idx]
            + self.eta * delta_t[idx] / (np.sqrt(self.v_t[idx]) + self.tau)
            for idx in range(len(delta_t))
        ]
        self.current_weights = new_weights
        self.set_weights(net, new_weights)
        if new_weights is not None:
            print(f"Saving round {rnd} model...")
            torch.save(net, f"round-{rnd}-model.pt")
            with open('contrib.json', 'w') as outfile:
                json.dump(self.contrib, outfile)

        return self.current_weights