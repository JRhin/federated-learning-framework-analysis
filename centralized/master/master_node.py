import os
import logging
import numpy as np
from pathlib import Path
import scipy.stats as ss
from collections import defaultdict
from model import LogisticRegression
from flask import Flask, request, jsonify

app = Flask(__name__)

class BayesianBootstrap:
    def __init__(self, concentration: float = 1.):
        self.n_draws = 100000
        self.bins = 100
        self.concentration = concentration


    def sample(self, obs: np.ndarray, weights: np.ndarray = None):

        # If no weights passed, use uniform Dirichlet
        if weights is None:
            weights = np.ones(len(obs))

        # Normalize weights to mean concentration
        weights = weights / weights.mean() * self.concentration

        # Sample posteriors
        draws = ss.dirichlet(weights).rvs(self.n_draws)
        means = (draws * obs).sum(axis=1)
        vars = draws * (obs - means.reshape(self.n_draws, 1)) ** 2

        return np.mean(means)


    def distribution(self, obs: np.ndarray, weights: np.ndarray = None):

        # Sample and create distribution objects
        means, stds = self.sample(obs, weights)
        hist_mean = np.histogram(means, bins=self.bins)
        hist_std = np.histogram(stds, bins=self.bins)

        return ss.rv_histogram(hist_mean), ss.rv_histogram(hist_std)




@app.route("/subscribe", methods=["GET"])
def subscribe():
    global id_client
    id_client += 1
    logger.info(f"A client asked the model, client {id_client} has started.")
    return jsonify({'id_client': id_client}), 200


@app.route("/push-weights", methods=["POST"])
def push_weights():
    try:
        global n_clients
        global weights
        global final_weights

        boot = BayesianBootstrap()
        n_clients -= 1
        data = request.json
        state_dict_json = data['model_state_dict']
        id_client = data['id_client']
        obs = data['obs']

        for weight in state_dict_json['linear.weight']:
            weights[weight]['value'].append(state_dict_json['linear.weight'][weight])
            weights[weight]['obs'].append(obs)

        weights['linear.bias']['value'].append(state_dict_json['linear.bias'])
        weights['linear.bias']['obs'].append(obs)

        logger.info(f'A client pushed the weights, client {id_client} has ended: {n_clients} clients active.')

        if n_clients == 0:
            for weight in weights:
                final_weights[weight] = boot.sample(np.array(weights[weight]['value']), np.array(weights[weight]['obs']))

            logger.info(final_weights)

        return f"State dict arrived from {id_client}.", 200

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return f"Error: {str(e)}", 400




if __name__ == "__main__":

    current = Path('.')
    logs = current/'logs'

    logs.mkdir(exist_ok=True)

    logging.basicConfig(filename=logs/"master_node.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

    logger = logging.getLogger()

    n_clients =  int(os.environ['CLIENTS'])
    id_client = -1

    def def_value():
        d = {'value': [],
             'obs': []}
        return d

    weights = defaultdict(def_value)
    final_weights = {}

    central_model = LogisticRegression(114, 1)

    logger.info("Starting the master node.")
    app.run(debug=True, host='0.0.0.0', port=80)
