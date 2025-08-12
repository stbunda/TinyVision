import argparse

import pytorch_lightning as pl
import torch
import torchvision.models as models
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data import get_dataset_class
from model.LightningModel import LightningModel, LightningModelLight

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    torch.backends.cudnn.benchmark = True

    print('############### Initializing Dataset ###############')
    dataset = get_dataset_class(args.dataset)(data_dir=args.data_dir)

    dataset.setup('fit')
    dataset.test_dataloader(batch_size=args.batch_size,
                            num_workers=16,
                            pin_memory=True)

    print('############### Creating model ###############')
    model = models.__dict__[args.model.lower()](pretrained=args.pretrained, num_classes=len(dataset.classes))
    model.to(device)
    model_name = args.model.lower() + f'_{args.width}'


    logger = TensorBoardLogger("lightning_logs", name=f"{model_name}-{args.dataset.lower()}", default_hp_metric=False)
    pl_model = LightningModelLight(model, datamodule=dataset)

    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                          filename=f'{model_name}-{args.dataset.lower()}'+'-{epoch}-{valid_loss:.2f}-{valid_acc:.2f}',
                                          save_top_k=3)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(accelerator='gpu' if args.device == 'cuda' else 'cpu', logger=logger, max_epochs=args.epochs, enable_progress_bar=False,
                         callbacks=[checkpoint_callback, lr_monitor])
    trainer.test(model=pl_model, datamodule=dataset, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--model', default='mobilenet_v3_small', help='model')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
    parser.add_argument('--data_dir', default='data', help='dataset directory')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('-e', '--epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')

    # Optimizer
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--random-erase', default=0.0, type=float, help='random erasing probability (default: 0.0)')

    parser.add_argument("-w",
                        "--width",
                        dest="width",
                        type=float,
                        help="Select width multiplier of model",
                        default=1.0)

    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=False,
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    arguments = parser.parse_args()
    print(f'arguments: {arguments}')
    main(arguments)