import torch
from torch import nn
import torch.nn.functional as F

import lightning as L

from .transform import DEPTH


class Simple(nn.Module):
    """..."""

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.full1 = nn.Linear(height * width * DEPTH, 256)
        self.full2 = nn.Linear(256, 512)
        self.dropout = nn.Dropout1d(0.5)
        self.policy_head = nn.Linear(512, width)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        h = torch.flatten(x, start_dim=1)
        h = F.leaky_relu(self.full1(h))
        h = F.leaky_relu(self.full2(h))
        h = self.dropout(h)
        policy_logits = self.policy_head(h)
        value_logits = self.value_head(h).squeeze(1)
        return policy_logits, value_logits


class ConnectModel(L.LightningModule):
    """..."""

    def __init__(
        self,
        height,
        width,
        model,
        learning_rate,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert model == "simple"
        self.model = Simple(height, width)

    def forward(self, x):
        policy_logits, value_logits = self.model(x)
        # TODO maybe normalize? this is used at inference only
        return policy_logits, value_logits

    def training_step(self, batch, batch_idx):
        x, y_policy, y_value = batch
        policy_logits, value_logits = self.model(x)
        policy_loss = F.kl_div(F.log_softmax(policy_logits, dim=-1), y_policy, reduction="batchmean")
        value_loss = F.binary_cross_entropy_with_logits(value_logits, y_value)
        # TODO weighted combination?
        loss = policy_loss + value_loss
        self.log("train_policy_loss", policy_loss)
        self.log("train_value_loss", value_loss)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])
