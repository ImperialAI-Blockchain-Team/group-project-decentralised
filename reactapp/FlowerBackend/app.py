import multiprocessing
from web3 import Web3
import abi
import requests
from flask import Flask, request, abort, Response, json, send_from_directory, jsonify
from flask_cors import CORS, cross_origin
from subprocess import call
import re, ast, json

########################################################################################################


web3 = Web3(Web3.HTTPProvider('https://ropsten.infura.io/v3/ec89decf66584cd984e5f89b6467f34f'))
job_contract_address = '0x65a6DCe3ce74b409Adb1B31CC53Cd6c141A8c681'
contract = web3.eth.contract(address=job_contract_address, abi=abi.job_abi)

# Tests
# print('Test:')
# job = contract.functions.jobs(0).call()
# strategy_hash = job[2]
# is_training_allowed = job[8]
# params = (('arg', strategy_hash),)
# response = requests.post('https://ipfs.infura.io:5001/api/v0/get', params=params)
# strategy = response.text
# strategy = re.search('{(.*)}', strategy).group(1)
# strategy = json.loads('{'+strategy+'}')
# strategy['minClients'] = int(strategy['minClients'])


########################################################################################################

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['TESTING'] = True

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

    # retrieve strategy from ipfs
    strategy_hash = job[2]
    params = (('arg', strategy_hash),)
    response = requests.post('https://ipfs.infura.io:5001/api/v0/get', params=params)
    strategy = response.text
    strategy = re.search('{(.*)}', strategy).group(1)
    strategy = json.loads('{'+strategy+'}')
    strategy['minClients'] = int(strategy['minClients'])

    # save strategy
    with open('data.json', 'w') as outfile:
        json.dump(strategy, outfile)

    # Start flower server
    call(["python", "server.py"])

    return {'server address': '[::]:8080'}, 200


if __name__ == "__main__":
    # app.run(debug=True)
    pass
