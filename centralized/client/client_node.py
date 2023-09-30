import json
import requests
import polars as pl
from pathlib import Path
#from model import LogisticRegression

def subscribe(url: str):

    response = requests.get(f'{url}/subscribe')

    model = 1 #LogisticRegression(10, 1)
    return model, response

def send_weights(state_dict,
                 obs: int) -> None:
    state_dict_json = {key: value.numpy().tolist() for key, value in state_dict.items()}
    return state_dict_json#json.dumps(state_dict_json)



if __name__ == "__main__":

    current = Path(".")
    data_path = current/"data"
    url = "http://master"

    # Get the model
    model, response = subscribe(url)

    id_client = response.json()['id_client']

    df = pl.read_csv(data_path/f"hospital_{id_client}.csv")
    print(df)
    
    
    # Train
    # ...

    # Send the weigths
    #send_weights(model.state_dict(), 3)
