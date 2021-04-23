import flwr as fl
from strategies.FedAvg import FedAvg
from strategies.FedOpt import FedOpt
from typing import Callable, Dict, List, Optional, Tuple
import uploads.model as ICU
import json
import torch
import os.path, requests
from app import contract, job_contract_address
import numpy as np
from collections import OrderedDict
from web3 import Web3

DEFAULT_SERVER_ADDRESS = "[::]:8080"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "uploads/testset.csv"

# Sign in to Ethereum Account
web3 = Web3(Web3.HTTPProvider('https://ropsten.infura.io/v3/ec89decf66584cd984e5f89b6467f34f'))
account = web3.eth.account.from_key('0x6b162e9dbfa762373e98b3944279f67b8fac61dc85f255da0108ebdc408af182')
web3.eth.default_account = account


def get_eval_fn(
    testset,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        model = ICU.Loader(DATA_ROOT).load_model()
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
        )
        model.load_state_dict(state_dict, strict=True)

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
            model=ICU
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
            model=ICU
        )

    return strategy

def launch_fl_server():
    strategy = configure_flower_server()
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": data["round"]}, strategy=strategy)

def contrib_cal(rnd):
    if os.path.exists('contrib.json'):
        f = open('contrib.json', )
        data = json.load(f)
        f.close()
        weights = []
        for i in range(rnd):
            weights.append(1/(i+1))
        m = 0
        for k, v in data.items():
            data[k]=np.average(v, weights=weights)
            m += data[k]
        for k, v in data.items():
            data[k] = v/m
        return data

    return 'No such file'

def calculate_compensations(job_id, compensation_weights):
    # Retrieve Job's bounty
    job = contract.functions.jobs(job_id).call()
    bounty = job[10]

    # Calculate compensations
    compensations = {address: int(weight*bounty) for address, weight in compensation_weights.items()}

    # Save compensations in the log ?

    return compensations

def save_weights_and_training_log():
    client = ipfshttpclient.connect('/dns/ipfs.infura.io/tcp/5001/https')
    model_weights_hash = client.add('./.model_weights.pt')['Hash']
    log_hash = client.add('./.log.json')['Hash']
    return model_weights_hash, log_hash


def send_compensations(job_id, compensations, model_weights_hash, log_hash):
    addresses = []
    compensation_values = []
    for address, value in compensations.items():
        addresses.append(address)
        compensation_values.append(value)
    receipt = contract.functions.compensate(job_id, compensation_values, addresses, model_weights_hash, log_hash).transact()
    return receipt




if os.path.exists('uploads/strategy.json'):
    f = open('uploads/strategy.json', )
    data = json.load(f)
    f.close()
    if data['strategy'] != "":
        launch_fl_server()
        with open('uploads/job_id.txt', 'r') as f:
            job_id = int(f.readline())

        compensation_weights = contrib_cal(data["round"])
        compensations = calculate_compensations(job_id, compensation_weights)
        model_weights_hash, log_hash = save_weights_and_training_log()
        send_compensations(job_id, compensations, model_weights_hash, log_hash)