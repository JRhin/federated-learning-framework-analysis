import json
import logging
import requests
import polars as pl
from pathlib import Path
from model import LogisticRegression

def subscribe(url: str):
    response = requests.get(f'{url}/subscribe')
    id_client = response.json()['id_client']
    logger.info(f"Subcribed to the Master Node. ID client assigned {id_client}.")
    return id_client

def preprocessing(data: pl.DataFrame):
    logger.info("Preprocessing performed.")
    return data

def send_weights(url: str,
                 state_dict,
                 obs: int) -> None:
    url = f'{url}/push-weights'
    state_dict_json = json.dumps({key: value.numpy().tolist() for key, value in state_dict.items()})
    response = requests.post(url, json={'model_state_dict': state_dict_json, 'obs': obs})
    logger.info("Weigths sent.")
    return response.status_code



if __name__ == "__main__":

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

    df = pl.read_csv(data_path/f"hospital_{id_client}.csv")

    # Preprocessing
    df = preprocessing(df)

    obs, features = df.shape
    features -= 1

    # Get the model
    model = LogisticRegression(features, 1)

    print(model)
    
    
    # Train
    # ...

    # Send the weigths
    status = send_weights(url, model.state_dict(), obs)
    print(status)
