from flask import Flask, request, abort, Response, json, send_from_directory, jsonify
from flask_cors import CORS, cross_origin
from subprocess import call

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['TESTING'] = True

@app.route('/flower', methods=['GET', 'POST'])
def flower_server():
    call(["python", "server.py"])

@cross_origin()
@app.route('/start', methods=['POST'])
def result():
    """Return a function which returns training configurations."""
    username = request.json.get("name","")
    data = {}
    data['name'] = username
    data['address'] = request.json.get("address","")
    data['strategy'] = request.json.get("strategy","")
    data['epoch'] = request.json.get("epoch", "")
    data['batch_size'] = request.json.get("batch_size", "")
    data['round'] = request.json.get("round", "")
    data['lr'] = request.json.get("lr", "")
    data['fraction_eval'] = request.json.get("fraction_eval", "")
    data['fraction_fit'] = request.json.get("fraction_fit", "")
    data['min_fit_clients'] = request.json.get("min_fit_clients", "")
    data['min_eval_clients'] = request.json.get("min_eval_clients", "")
    data['min_clients'] = request.json.get("min_clients", "")
    data['failure'] = request.json.get("failure'", "")
    data['beta'] = request.json.get("beta", "")
    data['slr'] = request.json.get("slr", "")
    data['clr'] = request.json.get("clr", "")
    data['da'] = request.json.get("da", "")
    data['distr'] = request.json.get("distr", "")
    data['mean'] = request.json.get("mean", "")
    data['std'] = request.json.get("std", "")
    data['ub'] = request.json.get("ub", "")
    data['lb'] = request.json.get("lb", "")
    data['gain'] = request.json.get("gain", "")
    data['fan'] = request.json.get("fan", "")
    data['linear'] = request.json.get("linear", "")
    data['slope'] = request.json.get("slope", "")

    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)
    return "Finish"



'''
###############################
##       for app.py          ##
###############################
import multiprocessing
from web3 import Web3
import abi
import requests

web3 = Web3(Web3.HTTPProvider('https://ropsten.infura.io/v3/ec89decf66584cd984e5f89b6467f34f'))
job_contract_address = '0x2F2dbC0ca7Bf1390196DCE21B595BDA29834B6C5'
contract = web3.eth.contract(address=job_contract_address, abi=abi.job_abi)

# Tests
# job = contract.functions.jobs(0).call()
# print(job)
# strategy_hash = 'QmSajo76z4JZmNwe7FXzongTYCUDYwJWhWXAvpjmr3uMCy'
# params = (('arg', strategy_hash),)
# response = requests.post('https://ipfs.infura.io:5001/api/v0/get', params=params)
# print(response.text)

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
    print(response.text)

    # create strategy

    # Start flower server
    flower_server_address = "[::]:8080"
    thread = multiprocessing.Process(target=start_flower_server, args=(flower_server_address, strategy))
    thread.start()
    return {"flower server address": flower_server_address}, 200


def start_flower_server(server_address, strategy):
    fl.server.start_server(server_address=server_address,
                            # strategy=strategy,
                            config={"num_rounds": 3})

if __name__ == "__main__":
    # app.run(debug=True)
    pass



'''