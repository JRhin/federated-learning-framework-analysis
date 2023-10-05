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
    logger.info("Start preprocessing...")
    data = data.drop(['encounter_id', 'icu_d', 'icu_stay_type', 'patient_id'])

    numerical_cat = [
        'elective_surgery',
        'apache_post_operative',
        'arf_apache',
        'gcs_unable_apache',
        'intubated_apache',
        'ventilated_apache',
        'aids',
        'cirrhosis',
        'diabetes_mellitus',
        'hepatic_failure',
        'immunosuppression',
        'leukemia',
        'lymphoma',
        'solid_tumor_with_metastasis']

    categorical = [
        'ethnicity',
        'gender',
        'icu_admit_source',
        'icu_type',
        'apache_3j_bodysystem',
        'apache_2_bodysystem']

    numeric_only = list(set(data.columns)-set(numerical_cat + categorical+['hospital_death', 'hospital_id']))

    # Subsitute the missing values:
    #  - categorical: mode
    #  - numerical: median
    data = data.with_columns(
            [pl.col(col).fill_null(data.get_column(col).mode().item()) for col in numerical_cat + categorical] +
            [pl.col(col).fill_null(pl.median(col)) for col in numeric_only]
            )

    data = data.with_columns(
            pl.col('gender').map_dict({'M':0, 'F':1})
            )

    categorical.remove('gender')
    data = data.to_dummies(columns=categorical, separator=':')
    return data


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

    # Preprocess the data
    df = preprocessing(df)
    logger.info("Done preprocessing.")

    obs, features = df.shape
    features -= 1

    # Get the model
    model = LogisticRegression(features, 1)

    # Train
    # ...

    # Send the weigths
    columns = df.columns
    columns.remove('hospital_death')
    status = send_weights(id_client,
                          obs,
                          model.state_dict(),
                          columns,
                          url)
    print(status)
