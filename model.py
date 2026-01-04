"""Custom models for the agentic demo with full ML workflow support."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DemoClassifier(pl.LightningModule):
    """A LightningModule with full workflow support: train, validate, test, predict."""

    def __init__(self, input_dim: int = 4, num_classes: int = 3, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Linear(input_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, _batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, _batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"test_loss": loss, "test_acc": acc}

    def predict_step(self, batch, _batch_idx: int):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return torch.argmax(self(x), dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _make_data(self, n: int = 64) -> TensorDataset:
        x = torch.randn(n, self.hparams.input_dim)
        y = torch.randint(0, self.hparams.num_classes, (n,))
        return TensorDataset(x, y)

    def train_dataloader(self):
        return DataLoader(self._make_data(64), batch_size=8)

    def val_dataloader(self):
        return DataLoader(self._make_data(32), batch_size=8)

    def test_dataloader(self):
        return DataLoader(self._make_data(32), batch_size=8)
