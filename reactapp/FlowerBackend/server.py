from flask import Flask, request
import flwr as fl
from flwr.server.strategy import FedAvg
from strategies.FedOpt import FedOpt
import multiprocessing
from web3 import Web3
import abi
import requests

web3 = Web3(Web3.HTTPProvider('https://ropsten.infura.io/v3/ec89decf66584cd984e5f89b6467f34f'))
job_contract_address = '0x2F2dbC0ca7Bf1390196DCE21B595BDA29834B6C5'
contract = web3.eth.contract(address=job_contract_address, abi=abi.job_abi)

# Tests
job = contract.functions.jobs(0).call()
print(job)
strategy_hash = 'QmSajo76z4JZmNwe7FXzongTYCUDYwJWhWXAvpjmr3uMCy'
params = (('arg', strategy_hash),)
response = requests.post('https://ipfs.infura.io:5001/api/v0/get', params=params)
print(vars(response))

app = Flask(__name__)

@app.route('/start_flower_server', methods=['GET'])
def configure_flower_server():

    # retrieve job id
    if 'id' not in request.args.keys():
        return {'error': 'id required'}, 500
    job_id = request.args['id']

    if not job_id.isdigit():
        return {'error': 'id must be a non negative integer'}, 500

    # verify job is allowed to be ran
    job = contract.functions.jobs(0).call()
    if not job[8]:
        return {'error': 'job cannot be started'}, 500

    # retrieve file from ipfs
    strategy_hash = job[2]
    params = (('arg', strategy_hash),)
    response = requests.post('https://ipfs.infura.io:5001/api/v0/get', params=params)
    print(response)

    # create strategy

    # Start flower server
    flower_server_address = "[::]:8080"
    thread = multiprocessing.Process(target=start_flower_server, args=(flower_server_address, strategy))
    thread.start()
    return {"flower server address": flower_server_address}


def start_flower_server(server_address, strategy):
    fl.server.start_server(server_address=server_address,
                            # strategy=strategy,
                            config={"num_rounds": 3})

if __name__ == "__main__":
    # app.run(debug=True)
    pass