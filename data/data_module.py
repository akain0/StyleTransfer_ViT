import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from data.dataset import StyleTransferDataset

def collate_fn(batch):
    """
    Stack style and content images into a batch.
    """
    style = torch.stack([sample["style_image"] for sample in batch])
    content = torch.stack([sample["content_image"] for sample in batch])
    return {"style": style, "content": content}

class StyleTransferDM(pl.LightningDataModule):
    """
    Custom data module for style transfer.
    """

    def __init__(
      self,
      style_dir,
      content_dir,
      num_styles_per_image=100,
      train_val_test_split=[0.5, 0.3, 0.2],
      batch_size=128,
      num_workers=4,
      prefetch_factor=2,
      pin_memory=True      
    ):
        """
        Init with data paths and loader configs.
        """
        super().__init__()
        self.style_dir = style_dir
        self.content_dir = content_dir
        self.num_styles_per_image = num_styles_per_image
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        """
        Split dataset into train/val/test.
        """
        full_ds = StyleTransferDataset(
            self.style_dir,
            self.content_dir,
            num_styles_per_image=self.num_styles_per_image
        )
        self.train_ds, self.val_ds, self.test_ds = random_split(
            full_ds,
            self.train_val_test_split
        )

    def train_dataloader(self):
        """
        Train loader.
        """
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 1,
            shuffle=True,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=False
        )

    def val_dataloader(self):
        """
        Validation loader.
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 1,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=False
        )

    def test_dataloader(self):
        """
        Test loader.
        """
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.num_workers > 1,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=False
        )
