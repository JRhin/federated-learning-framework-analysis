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

    url = "http://master"

    # Get the model

    #while True:
    #    pass


    response = subscribe(url)
    
    print(response.text, "DONE")
    # Train
    # ...

    # Send the weigths
    #send_weights(model.state_dict(), 3)
