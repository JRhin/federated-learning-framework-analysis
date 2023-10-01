import json
import logging
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from model import LogisticRegression

app = Flask(__name__)

@app.route("/subscribe", methods=["GET"])
def subscribe():
    global n_clients 
    id_client = n_clients
    n_clients += 1
    logger.info(f"A client asked the model, {n_clients} clients active.")
    return jsonify({'id_client': id_client}), 200


@app.route("/push-weights", methods=["POST"])
def push_weights():
    global n_clients
    n_clients -= 1
    data = request.json
    state_dict_json = data['model_state_dict']
    obs = data['obs']
    logger.info(f'A client pushed the weights, {n_clients} clients active.')
    return state_dict_json, obs





if __name__ == "__main__":

    current = Path('.')
    logs = current/'logs'

    logs.mkdir(exist_ok=True)

    logging.basicConfig(filename=logs/"master_node.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    logger = logging.getLogger()

    n_clients =  0
    central_model = LogisticRegression(83, 1)

    logger.info("Starting the master node.")
    app.run(debug=True, host='0.0.0.0', port=80)
