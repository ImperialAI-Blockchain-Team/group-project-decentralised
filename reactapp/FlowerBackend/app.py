import requests
from abi import abi
from flask import Flask, request, abort, Response, json, send_from_directory, jsonify
from flask_cors import CORS, cross_origin
import subprocess
import re, json
import json
from web3 import Web3

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['TESTING'] = True
app.config['CORS_ORIGIN_ALLOW_ALL'] = True
app.config['ALLOWED_HOSTS'] = ['*']
abi = json.loads(abi)

web3 = Web3(Web3.HTTPProvider('https://ropsten.infura.io/v3/ec89decf66584cd984e5f89b6467f34f'))
account = web3.eth.account.from_key('0x6b162e9dbfa762373e98b3944279f67b8fac61dc85f255da0108ebdc408af182')
web3.eth.default_account = account._address
job_contract_address = '0xD1a210292F6D37098114AFF851D747Ba6ccBAB9B'
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
    if 'id' not in request.json:
        return {'error': 'id required'}, 500
    job_id = request.json['id']
    if not job_id.isdigit():
        return {'error': 'id must be a non negative integer'}, 500

    int_job_id = int(job_id)
    print(request.json['id'])
    # verify job is allowed to be ran
    job = contract.functions.jobs(int_job_id).call()
    print(job[8])
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
    subprocess.call(["python3", "server.py"])

    return {'server address': '[::]:8080'}, 200

def send_compensations(job_id, compensations, model_weights_hash, log_hash):
    addresses = []
    compensation_values = []
    for address, value in compensations.items():
        addresses.append(address)
        compensation_values.append(value)
    #  compensate(uint _id, uint [] memory _compensation, address payable [] memory _clients,
    #                         string memory _resultsHash, string memory _weightsHash)
    receipt = contract.functions.compensate(job_id, compensation_values, addresses, model_weights_hash, log_hash).transact()
    return receipt

if __name__ == "__main__":

    # app.run(debug=True)
    #start_server()
    job = contract.functions.jobs(0).call()
    print(job)
    jobResult = contract.functions.jobResults(0).call()
    print(jobResult)
    numAllow = contract.functions.getNumAllow(0).call()
    print(numAllow)
    #owner = job[0]
    #balance = web3.eth.get_balance(account._address)

    #receipt = contract.functions.compensate(int(0), [int(20000)], ['0xdeaA6Fa2395422b8f2919E14f927102Ea83Fc4Fb'], 'model_weights_hash','log_hash').transact()
    transaction = contract.functions.compensate(0,[20000],['0x030a713342D37EeBD51557aA56bc7Fe580A3910B'], 'hash', 'hash')\
        .buildTransaction({'chainId': 3, 'gas': 7000000, 'nonce': web3.eth.getTransactionCount(account._address)})
    signed_txn = web3.eth.account.signTransaction(transaction, account._private_key)
    print(transaction)
    print(signed_txn)
    txn_hash = web3.eth.sendRawTransaction(signed_txn.rawTransaction)
    print(txn_hash)
    receipt = web3.eth.wait_for_transaction_receipt(txn_hash)
    print('RECEIPT')
    print(receipt)
    print(receipt['status'])
    if receipt['status'] == 0:
        print('transaction failed')
    else:
        print('transaction success')
    #print(transaction)
    #print(owner)
    #print(numAllow)
    #print(account)
    #print(account._address)
    #print(account._private_key)

