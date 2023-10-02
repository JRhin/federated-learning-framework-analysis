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
    global id_client
    id_client += 1
    n_clients += 1
    logger.info(f"A client asked the model, {n_clients} clients active.")
    return jsonify({'id_client': id_client}), 200


@app.route("/push-weights", methods=["POST"])
def push_weights():
    try:
        global n_clients
        n_clients -= 1
        logger.info(request.json)
        data = request.json
        state_dict_json = data['model_state_dict']
        obs = data['obs']
        id_client = data['id_client']
        logger.info(f'A client pushed the weights, {n_clients} clients active.')
        return f"State dict arrived from {id_client}.", 200
    except Exception as e:
        return f"Error: {str(e)}", 400




if __name__ == "__main__":

    current = Path('.')
    logs = current/'logs'

    logs.mkdir(exist_ok=True)

    logging.basicConfig(filename=logs/"master_node.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    logger = logging.getLogger()

    n_clients =  0
    id_client = -1
    central_model = LogisticRegression(83, 1)

    logger.info("Starting the master node.")
    app.run(debug=True, host='0.0.0.0', port=80)
