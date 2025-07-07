import argparse

import pytorch_lightning as pl
import torch
import torchvision.models as models
from pytorch_lightning.loggers import TensorBoardLogger

from data.pytorch_dataset import CIFAR10
from model.LightningModel import LightningModel

def main(args):

    # Metadata:
    model_name = 'MobileNetV3S'
    data_set = 'CIFAR10'

    logger = TensorBoardLogger("lightning_logs", name=f"{model_name}_{data_set}", default_hp_metric=False)

    model = models.mobilenet_v3_small(num_classes=10, width_mult=args.width)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CIFAR10(data_dir='data',
                      train_val_split=0.2,
                      # subset=100
                      )

    dataset.setup('fit')
    dataset.train_dataloader(batch_size=128,
                             shuffle=True,
                             num_workers=8,
                             drop_last=True,
                             pin_memory=True)
    dataset.test_dataloader(batch_size=128,
                            num_workers=8,
                            pin_memory=True)

    pl_model = LightningModel(model, learning_rate=args.lr, datamodule=dataset)
    # pl_model.to(device)
    # pl_model.eval()

    # Evaluate model
    total_time = 0
    correct = 0
    total = 0

    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, enable_progress_bar=False)
    trainer.fit(model=pl_model, datamodule=dataset, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w",
                        "--width",
                        dest="width",
                        type=float,
                        help="Select width multiplier of model",
                        default=1.0)
    parser.add_argument("-e",
                        "--epochs",
                        dest="epochs",
                        type=float,
                        help="Select max epochs",
                        default=100)
    parser.add_argument("-l",
                        "--lr",
                        dest="lr",
                        type=float,
                        help="Select learning rate",
                        default=0.001)

    arguments = parser.parse_args()
    print(f'arguments: {arguments}')
    main(arguments)