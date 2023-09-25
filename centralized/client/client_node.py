import json
import requests
#from model import LogisticRegression

def subscribe(url: str):

    response = requests.get(f'{url}/subscribe')

    model = 1 #LogisticRegression(10, 1)
    return response

def send_weights(state_dict,
                 obs: int) -> None:
    state_dict_json = {key: value.numpy().tolist() for key, value in state_dict.items()}
    return state_dict_json#json.dumps(state_dict_json)



if __name__ == "__main__":

    url = "http://192.168.92.20:80"

    # Get the model
    response = subscribe(url)
    
    print(response)
    # Train
    # ...

    # Send the weigths
    #send_weights(model.state_dict(), 3)
