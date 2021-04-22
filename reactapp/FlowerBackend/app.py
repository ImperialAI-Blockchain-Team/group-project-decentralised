from web3 import Web3
import requests
from abi import abi
from flask import Flask, request, abort, Response, json, send_from_directory, jsonify
from flask_cors import CORS, cross_origin
import subprocess
import re, json
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['TESTING'] = True
abi = json.loads(abi)

web3 = Web3(Web3.HTTPProvider('https://ropsten.infura.io/v3/ec89decf66584cd984e5f89b6467f34f'))
job_contract_address = '0x1784f9C5b53888F07cFAeFEd8DD0C4ED4F2E60FF'
contract = web3.eth.contract(address=job_contract_address, abi=abi)

def retrieve_strategy(strategy_hash):
    params = (('arg', strategy_hash),)
    response = requests.post('https://ipfs.infura.io:5001/api/v0/get', params=params)
    strategy = response.text
    strategy = re.search('{(.*)}', strategy).group(1)
    strategy = json.loads('{'+strategy+'}')
    strategy['minClients'] = int(strategy['minClients'])
    with open('uploads/strategy.json', 'w') as outfile:
        json.dump(strategy, outfile)

def retrieve_model(model_hash):
    params = (('arg', model_hash),)
    response = requests.post('https://ipfs.infura.io:5001/api/v0/get', params=params)
    content = response.text
    content = content.split('\n')
    content = '\n'.join(content[1:-1])
    with open('uploads/model.py', 'w') as f:
        f.write(content)

def retrieve_testset(testset_hash):
    print('Test:')
    params = (('arg', testset_hash),)
    response = requests.post('https://ipfs.infura.io:5001/api/v0/get', params=params)
    content = response.text
    content = content.split('\n')
    content = '\n'.join(content[1:-1])
    with open('uploads/testset.csv', 'w') as f:
        f.write(content)

@cross_origin()
@app.route('/start_server', methods=['GET', 'POST'])
def start_server():
    # retrieve job id
    if 'id' not in request.args.keys():
        return {'error': 'id required'}, 500
    job_id = request.args['id']

    if not job_id.isdigit():
        return {'error': 'id must be a non negative integer'}, 500

    # verify job is allowed to be ran
    job = contract.functions.jobs(job_id).call()
    if not job[8]:
        return {'error': 'job cannot be started'}, 500

    # retrieve strategy, model and testset from ipfs and save them locally
    model_hash = job[1]
    strategy_hash = job[2]
    testset_hash = job[3]
    retrieve_model(model_hash)
    retrieve_strategy(strategy_hash)
    retrieve_testset(testset_hash)
    with open('uploads/job_id.txt', 'w') as f:
        f.write(job_id)

    # Start flower server
    subprocess.Popen(["python3", "server.py"])

    return {'server address': '[::]:8080'}, 200

if __name__ == "__main__":
    # app.run(debug=True)
    start_server()
    job = contract.functions.jobs(0).call()
    hash = job[2]
    retrieve_strategy(hash)
