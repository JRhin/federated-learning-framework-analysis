import gdown
import zipfile
from dotenv import dotenv_values
from pathlib import Path

def main():
    current = Path('.')
    env_path = current/'.env'
    data_path = current/'data'
    zip_path = data_path/'data.zip'

    data_path.mkdir(exist_ok=True)

    env = dotenv_values(env_path)

    gdown.download(id=env['ID'], output=str(zip_path))

    with zipfile.ZipFile(str(zip_path),"r") as zip_ref:
        zip_ref.extractall(data_path)

    zip_path.unlink()

if __name__ == "__main__":
    main()
