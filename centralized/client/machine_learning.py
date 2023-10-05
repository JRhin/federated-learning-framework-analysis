"""Python Module in which we define all the need classes for our machine-learning routine.
"""

# ===================================
#            Libraries
# ===================================

import torch
import polars as pl
from pathlib import Path
from torch import optim, nn
import lightning.pytorch as lg
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
                 data_path: Path,
                 id_client: int = None,
                 batch_size: int = 64,
                 num_workers: int = 0):
        super().__init__()
        self.id_client = id_client
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.features = []
        self.n_features = 0
        self.obs = 0


    def prepare_data(self) -> None:
       pass

    def setup(self,
              stage = None,
              seed: int = None) -> None:
        """In this function we setup the dataset.

        Args:
            - seed (int): The random seed for the torch.Generator. Default None.

        Returns:
            - None
        """
        csv_path = self.data_path/f'hospital_{self.id_client}.csv'

        if seed:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        data = pl.read_csv(csv_path).drop(['encounter_id', 'icu_d', 'icu_stay_type', 'patient_id', 'hospital_id'])

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

        numeric_only = list(set(data.columns)-set(numerical_cat + categorical+['hospital_death']))

        # Subsitute the missing values:
        #  - numerical_cat & categorical: mode
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

        columns = data.columns
        columns.remove('hospital_death')

        self.features = columns
        self.obs, self.n_features = data.shape
        self.n_features -= 1

        dataset = CustomDataSet(data, columns, 'hospital_death')
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

    datamodule = DataModule(data_path, num_workers=12)
    datamodule.prepare_data()
    datamodule.setup(seed=seed)

    model = LogisticRegression(datamodule.n_features)
    clf = Classifier(model, lr=1e-1)

    trainer = lg.Trainer(max_epochs=30)
    trainer.fit(clf, datamodule=datamodule)

    trainer.test(ckpt_path='best', datamodule=datamodule)
