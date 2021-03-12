# Requirements

## Install Poetry (https://python-poetry.org/docs/)
osx / linux / bashonwindows install instructions \
`curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`

windows powershell install instructions \
`(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -`

## Install Requirements with Poetry
`poetry install`


# Run Code

Start flask server

`cd fl_network/server; python3 app.py`

Data Scientist uploads a model to flask server

`cd fl_network/data_scientist; python3 upload_model.py`

Start Federated Learning Server and Training configuration

`cd fl_network/server; python3 server.py`

Download Model from flask server

`cd fl_network/client_1; python3 retrieve_model.py` \
`cd fl_network/client_2; python3 retrieve_model.py` \
`cd fl_network/client_3; python3 retrieve_model.py` 

Start Training

`cd fl_network/client_1; python3 client.py --cid 01` \
`cd fl_network/client_2; python3 client.py --cid 02` \
`cd fl_network/client_3; python3 client.py --cid 03`
