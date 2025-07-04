import torch
import torchvision.models as models
import pytorch_lightning as pl

from data.pytorch_dataset import CIFAR10
from model.LightningModel import LightningModel

model = models.mobilenet_v3_small()

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")



# Load dataset
dataset = CIFAR10(data_dir='data',
                    train_val_split=0.2,
                    # subset=100
                    )

dataset.setup('fit')
dataset.train_dataloader(batch_size=128,
                            shuffle=True,
                            drop_last=True,
                            num_workers=8,
                            pin_memory=True)
dataset.test_dataloader(batch_size=128,
                        num_workers=8,
                        pin_memory=True)

pl_model = LightningModel(model, learning_rate=0.001, datamodule=dataset)
pl_model.to(device)
pl_model.eval()

# Evaluate model
total_time = 0
correct = 0
total = 0

trainer = pl.Trainer()
trainer.fit(model=pl_model, datamodule=dataset)

