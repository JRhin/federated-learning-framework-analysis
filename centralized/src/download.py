import gdown
import zipfile
import polars as pl
from pathlib import Path
from tqdm.auto import tqdm
from dotenv import dotenv_values

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

    t = 100
    print(f'Save only hospital with more than {t=} observations:')
    for i, data in enumerate(tqdm(df.partition_by('hospital_id'))):
        if data.shape[0] >= t:
            data.write_csv(f'{str(data_path)}/hospital_{i}.csv')

    hp.unlink()


if __name__ == "__main__":
    main()
