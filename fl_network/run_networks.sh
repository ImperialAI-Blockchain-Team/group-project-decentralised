#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" # dir of bash script

cd $DIR/server; python3 app.py &
sleep 1 # waiting for the server to start

cd $DIR/data_scientist; python3 upload_model.py
cd $DIR/client_1; python3 retrieve_model.py
cd $DIR/client_2; python3 retrieve_model.py
cd $DIR/client_3; python3 retrieve_model.py

cd $DIR/server; python3 server.py &

cd $DIR/client_1/retrieved_models; python3 client.py &
cd $DIR/client_2/retrieved_models; python3 client.py &
cd $DIR/client_3/retrieved_models; python3 client.py &
