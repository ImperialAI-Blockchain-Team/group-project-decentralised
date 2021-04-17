import flwr as fl
from strategies.FedAvg import FedAvg
from strategies.FedOpt import FedOpt
import multiprocessing
import flwr as fl
from typing import Callable, Dict, List, Optional, Tuple
import uploads.model as ICU
import json
import torch
import os.path

DEFAULT_SERVER_ADDRESS = "[::]:8080"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "data/patient.csv"

def get_eval_fn(
    testset,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""
        model = ICU.Loader(DATA_ROOT).load_model()
        model.set_weights(weights)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

        return ICU.test(model, testloader, device=DEVICE)

    return evaluate
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""

        config = {
            "learning_rate": str(data['lr']),
            "batch_size": str(data["batch_size"]),
            "epochs": str(data["epoch"]),
        }
        return config

    return fit_config

def configure_flower_server():

    strategy_type = data["strategy"]
    if (strategy_type == 'datasize' or strategy_type == 'accuracy'):
        strategy = FedAvg(
            fraction_fit=data["fraction_fit"],
            fraction_eval=data["fraction_eval"],
            min_fit_clients=data["min_fit_clients"],
            min_available_clients=data["min_clients"],
            on_fit_config_fn=get_on_fit_config_fn(),
            eval_fn=None,
            on_evaluate_config_fn=None,
            accept_failures=data["failure"],
            mode=strategy_type,
            # eval_fn=get_eval_fn(testloader)
            # Minimum number of clients that need to be connected to the server before a training round can start
        )
    else:
        model = ICU.Loader(DATA_ROOT).load_model()
        distribution = data["distr"]
        for name, param in model.named_parameters():
            if distribution == 'normal':
                torch.nn.init.normal_(param, mean=float(data["mean"]), std=float(data["std"]))
            elif distribution == 'uniform':
                torch.nn.init.uniform_(param, a=float(data["lb"]), b=float(data["ub"]))
            else:
                if 'bias' not in name:
                    if distribution == 'xuniform':
                        torch.nn.init.xavier_uniform_(param, gain=float(data["gain"]))
                    if distribution == 'xnormal':
                        torch.nn.init.xavier_normal_(param, gain=float(data["gain"]))
                    if distribution == 'kuniform':
                        torch.nn.init.kaiming_uniform_(param, a=float(0), mode=data["fan"], nonlinearity=data["linear"])
                    if distribution == 'knormal':
                        torch.nn.init.kaiming_normal_(param, a=float(0), mode=data["fan"], nonlinearity=data["linear"])
                else:
                    torch.nn.init.zeros_(param)
        strategy = FedOpt(
            fraction_fit=data["fraction_fit"],
            fraction_eval=data["fraction_eval"],
            min_fit_clients=data["min_fit_clients"],
            min_available_clients=data["min_clients"],
            on_fit_config_fn=get_on_fit_config_fn(),
            eval_fn=None,
            on_evaluate_config_fn=None,
            accept_failures=data["failure"],
            beta=data["beta"],
            initial_parameters=model.get_weights(),
            eta=data["slr"],
            eta_l=data["clr"],
            tau=data["da"],
            mode=strategy_type,
        )

    return strategy


def launch_fl_server():
    strategy = configure_flower_server()
    fl.server.start_server(DEFAULT_SERVER_ADDRESS, config={"num_rounds": data["round"]}, strategy=strategy)

if os.path.exists('data.json'):
    f = open('data.json', )
    data = json.load(f)
    f.close()
    if data['name'] != "":
        launch_fl_server()