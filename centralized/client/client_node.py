import json
import requests
from model import LogisticRegression

def get_model(url: str) -> LogisticRegression:

    response = requests.get(f'{url}/get-model')

    model = LogisticRegression(10, 1)
    return model

def send_weights(state_dict,
                 obs: int) -> None:
    state_dict_json = {key: value.numpy().tolist() for key, value in state_dict.items()}
    return state_dict_json#json.dumps(state_dict_json)



if __name__ == "__main__":

    url = "http://127.0.0.1:5000"

    # Get the model
    model = get_model(url)
    
    # Train
    # ...

    # Send the weigths
    send_weights(model.state_dict(), 3)
