from flask import Flask, request
import flwr as fl
from flwr.server.strategy import FedAvg
from strategies.FedOpt import FedOpt
import multiprocessing

app = Flask(__name__)

@app.route('/start_flower_server', methods=['GET'])
def configure_flower_server():
    # Configure server strategy
    strategy_params = request.args.to_dict()
    strategy_type = strategy_params['type']
    del strategy_params['type']
    if strategy_type == 'FedAvg':
        strategy = FedAvg(*strategy_params)
    elif strategy_type == 'FedOpt':
        strategy = FedOpt(*strategy_params)
    strategy = None

    # Start flower server with the given strategy
    flower_server_address = "[::]:8080"
    thread = multiprocessing.Process(target=start_flower_server, args=(flower_server_address, strategy))
    thread.start()
    return {"flower server address": flower_server_address}


def start_flower_server(server_address, strategy):
    fl.server.start_server(server_address=server_address,
                            # strategy=strategy,
                            config={"num_rounds": 3})


if __name__ == "__main__":
    app.run(debug=True)