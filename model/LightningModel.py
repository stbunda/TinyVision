import pytorch_lightning as pl
import torch
import torchmetrics
from tensorflow.python.layers.core import dropout

# LightningModule that receives a PyTorch model as input
class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate, datamodule, optimizer, lr_scheduler):
        super().__init__()

        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.model = model
        if hasattr(model, "dropout_proba"):
            self.dropout_proba = model.dropout_proba

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        # self.save_hyperparameters(ignore=["model"])

        self.datamodule = datamodule
        self.given_optimizer = optimizer
        self.given_lr_scheduler = lr_scheduler

        # Set up attributes for computing the accuracy
        self.train_acc = torchmetrics.Accuracy(task=datamodule.task, num_classes=len(datamodule.classes))
        self.valid_acc = torchmetrics.Accuracy(task=datamodule.task, num_classes=len(datamodule.classes))
        self.test_acc = torchmetrics.Accuracy(task=datamodule.task, num_classes=len(datamodule.classes))

    # Defining the forward method is only necessary
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)

    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)

        # Do another forward pass in .eval() mode to compute accuracy
        # while accountingfor Dropout, BatchNorm etc. behavior
        # during evaluation (inference)
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_step(batch)
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.model.train()

        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.valid_acc(predicted_labels, true_labels)
        self.log(
            "valid_acc",
            self.valid_acc,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        opt = self.given_optimizer
        scheduler = self.given_lr_scheduler

        return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'monitor': 'val_loss'}}