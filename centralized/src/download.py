import gdown
import zipfile
import polars as pl
from pathlib import Path
from tqdm.auto import tqdm
from dotenv import dotenv_values

def preprocessing(data: pl.DataFrame):
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

def main():
    current = Path('.')
    client = current/'client'
    env_path = current/'.env'
    data_path = client/'data'
    zip_path = data_path/'data.zip'

    data_path.mkdir(exist_ok=True)

    env = dotenv_values(env_path)

    gdown.download(id=env['ID'], output=str(zip_path))

    with zipfile.ZipFile(str(zip_path),"r") as zip_ref:
        zip_ref.extractall(data_path)

    zip_path.unlink()

    hp = data_path/'hospital_dataset.csv'
    df = pl.read_csv(hp, dtypes={'d1_sysbp_noninvasive_min': pl.Float32, 'resprate_apache': pl.Float32})
    df = df.drop('')

    df = preprocessing(df)

    t = 100
    print(f'Save only hospital with more than {t=} observations:')
    for i, data in enumerate(tqdm(df.partition_by('hospital_id'))):
        if data.shape[0] >= t:
            data.drop('hospital_id').write_csv(f'{str(data_path)}/hospital_{i}.csv')

    hp.unlink()


if __name__ == "__main__":
    main()
