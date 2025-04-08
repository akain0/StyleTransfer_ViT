import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Lightning Data Module
class MyCustomDataModule(pl.LightningDataModule):
    def __init__(
      self,
      data_dir: str = None,
      batch_size: int = 32
    ):
        """
        Args:
            data_dir (str): Directory where the data is located.
            batch_size (int): Size of each data batch.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        """
        Set up the dataset for different stages (train, validation, test).
        This method is called by Lightning with the proper stage.
        """
        # Create dataset instances for training, validation, and testing
        self.train_dataset = MyDataset(data_path=self.data_dir)
        self.val_dataset = MyDataset(data_path=self.data_dir)
        self.test_dataset = MyDataset(data_path=self.data_dir)

    def train_dataloader(self):
        return DataLoader(
          self.train_dataset,
          batch_size=self.batch_size,
          shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
          self.val_dataset,
          batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
          self.test_dataset,
          batch_size=self.batch_size
        )
