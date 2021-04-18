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




