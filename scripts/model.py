import torch.nn as nn
import pytorch_lightning as pl

class MyLightningModel(pl.LightningModule):
    def __init__(
      self,
      input_dim: int = 10,
      output_dim: int = 2
    ):
        super().__init__()
        # Define your model architecture here.
        self.layer = nn.Linear(input_dim, output_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Define forward propagation
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        # Unpack the batch (assuming a tuple (inputs, targets))
        x, y = batch
        y_hat = self.forward(x)
        # Compute training loss (using cross-entropy as an example)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch and compute validation loss
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # Unpack the batch and get predictions
        x, y = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        # Define and return your optimizer(s)
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
