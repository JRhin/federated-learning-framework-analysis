import json
import logging
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify
from model import LogisticRegression

app = Flask(__name__)

@app.route("/get-model", methods=["GET"])
def get_model():
    global n_clients 
    global central_model
    n_clients += 1
    logger.info(f"A client asked the model, {n_clients} clients active.")

    model_schema = []
    
    for name, module in central_model.named_children():
        layer_info = {
                    'type': module.__class__.__name__,
                    'parameters': [p for p in module.parameters()]
                }
        model_schema.append(layer_info)

    print(str(central_model))
    print(model_schema)

    return jsonify({'model_schema': model_schema}), 200


@app.route("/push-weights", methods=["POST"])
def push_weights():
    global n_clients
    n_clients -= 1
    logger.info(f'A client pushed the weights, {n_clients} clients active.')
    pass





if __name__ == "__main__":

    current = Path('.')
    logs = current/'logs'

    logs.mkdir(exist_ok=True)

    logging.basicConfig(filename=logs/"master_node.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    logger = logging.getLogger()

    n_clients =  0
    central_model = LogisticRegression(10, 1)

    logger.info("Starting the master node.")
    app.run(debug=True)
