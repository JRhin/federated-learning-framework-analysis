"""Python Module in which we define all the need classes for our machine-learning routine.
"""

# ===================================
#            Libraries
# ===================================

import gdown
import torch
import zipfile
import polars as pl
from pathlib import Path
from torch import optim, nn
import lightning.pytorch as lg
from dotenv import dotenv_values
from model import LogisticRegression
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

# ===================================
#               Classes
# ===================================

class CustomDataSet(Dataset):
    def __init__(self,
                 dataframe: pl.DataFrame,
                 columns: list[str],
                 target: str):
        self.columns = columns
        self.target = target
        self.dataframe = dataframe.select(self.columns+[self.target])


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        row = self.dataframe.row(idx, named=True)
        input_tokens = [row[col] for col in self.columns]
        label = row[self.target]
        return torch.tensor(input_tokens), torch.tensor(label, dtype=torch.float)



class DataModule(lg.LightningDataModule):
    def __init__(self,
                 id: str,
                 data_path: Path,
                 batch_size: int = 64,
                 num_workers: int = 0):
        super().__init__()
        self.done = False
        self.id = id
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.features = 0


    def prepare_data(self) -> None:
        """We create the data folder (if not exists) and than download,
        the csv file and unzip it.

        Returns:
            - None
        """
        if not self.done:
            self.data_path.mkdir(exist_ok=True)

            zip_path = self.data_path/'data.zip'
            gdown.download(id=self.id, output=str(zip_path))

            with zipfile.ZipFile(file=str(zip_path), mode='r')as zip_ref:
                zip_ref.extractall(self.data_path)

            zip_path.unlink()
            self.done = True


    def setup(self,
              stage = None,
              seed: int = None) -> None:
        """In this function we setup the dataset.

        Args:
            - seed (int): The random seed for the torch.Generator. Default None.

        Returns:
            - None
        """
        csv_path = self.data_path/'*.csv'
        map_d = {'Y': 1, 'N': 0, 'F': 0, 'M': 1}
        if seed:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        df = pl.read_csv(csv_path)
        df = df.drop('ID').with_columns(pl.col('gender').map_dict(map_d),
                                        pl.col('oral').map_dict(map_d),
                                        pl.col('tartar').map_dict(map_d))

        columns = df.columns
        columns.remove('smoking')

        self.features = len(columns)

        dataset = CustomDataSet(df, columns, 'smoking')
        self.training_data, self.test_data, self.validation_data = random_split(dataset, [0.8, 0.1, 0.1], generator)


    def train_dataloader(self) -> DataLoader:
        """The function used to return the train DataLoader.

        Returns:
            -  DataLoader : the train DataLoader.
        """
        return DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)


    def val_dataloader(self) -> DataLoader:
        """The function used to return the validation DataLoader.

        Returns:
            -  DataLoader : the validation DataLoader.
        """
        return DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


    def test_dataloader(self) -> DataLoader:
        """The function used to return the test DataLoader.

        Returns:
            -  DataLoader : the test DataLoader.
        """
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



class Classifier(lg.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr


    def training_step(self,
                      batch,
                      batch_idx: int):
        x, y = batch
        y_hat = self.model(x).squeeze(dim=1)

        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(y_hat, y)
        self.log("training_loss", loss)
        return loss


    def test_step(self,
                  batch,
                  batch_idx: int):
        x, y = batch
        y_hat = self.model(x).squeeze(dim=1)

        criterion = nn.BCEWithLogitsLoss()
        test_loss = criterion(y_hat, y)

        self.log("test_loss", test_loss)
        return test_loss


    def validation_step(self,
                        batch,
                        batch_idx: int):
        x, y = batch
        y_hat = self.model(x).squeeze(dim=1)

        criterion = nn.BCEWithLogitsLoss()
        val_loss = criterion(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss


    def configure_optimizers(self):
        """Function in which we configure the optimizer. We decided to use the Adam optimizer.
        """
        return optim.Adam(self.parameters(), lr=self.lr)




if __name__ == "__main__":
    seed = 42

    current = Path('.')
    data_path = current/'data'
    env_path = current/'.env'

    env = dotenv_values(env_path)
    id = env['ID']

    datamodule = DataModule(id, data_path, num_workers=12)
    datamodule.prepare_data()
    datamodule.setup(seed=seed)

    model = LogisticRegression(datamodule.features)
    clf = Classifier(model, lr=1e-1)

    trainer = lg.Trainer(max_epochs=30)
    trainer.fit(clf, datamodule=datamodule)

    trainer.test(ckpt_path='best', datamodule=datamodule)
