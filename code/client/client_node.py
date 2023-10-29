import logging
import requests
import polars as pl
from pathlib import Path
from model import LogisticRegression
from machine_learning import *

def subscribe(url: str):
    response = requests.get(f'{url}/subscribe')
    id_client = response.json()['id_client']
    logger.info(f"Subcribed to the Master Node. ID client assigned {id_client}.")
    return id_client


def send_weights(id_client: int,
                 obs: int,
                 state_dict,
                 columns: list[str],
                 url: str) -> None:
    url = f'{url}/push-weights'
    
    state_dict_json = {key: value.numpy().tolist() for key, value in state_dict.items()}

    weights = {}
    for i, w in enumerate(state_dict_json['linear.weight'][0]):
        weights[columns[i]] = w

    state_dict_json['linear.weight'] = weights
    state_dict_json['linear.bias'] = state_dict_json['linear.bias'][0] 
    
    response = requests.post(url, json={'model_state_dict': state_dict_json, 'obs': obs, 'id_client': id_client})
    logger.info("Weigths sent.")
    return response.status_code



if __name__ == "__main__":

    seed = 42
    current = Path(".")
    logs = current/"logs"
    data_path = current/"data"
    url = "http://master"

    # Logging stuff
    logs.mkdir(exist_ok=True)
    logging.basicConfig(filename=logs/"client_node.logs", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    logger = logging.getLogger()


    # Subscribe to the Master Node
    id_client = subscribe(url)

    datamodule = DataModule(data_path, id_client=id_client, num_workers=12)
    datamodule.setup(seed=seed)

    model = LogisticRegression(datamodule.n_features)
    clf = Classifier(model, lr=1e-1)

    # Training
    trainer = lg.Trainer(max_epochs=30)
    trainer.fit(clf, datamodule=datamodule)

    # Test
    trainer.test(ckpt_path='best', datamodule=datamodule)


    # Send the weigths
    status = send_weights(id_client,
                          datamodule.obs,
                          model.state_dict(),
                          datamodule.features,
                          url)
    print(status)
