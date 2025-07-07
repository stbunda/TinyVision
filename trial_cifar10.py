import argparse

import pytorch_lightning as pl
import torch
import torchvision.models as models
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data.pytorch_dataset import CIFAR10
from model.LightningModel import LightningModel

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    if args.dataset.lower() == 'cifar10':
        dataset = CIFAR10(data_dir='data',
                          train_val_split=0.2,
                          # subset=100
                          )
    else:
        NotImplementedError()
    dataset.setup('fit')
    dataset.train_dataloader(batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=16,
                             drop_last=True,
                             pin_memory=True)
    dataset.test_dataloader(batch_size=args.batch_size,
                            num_workers=16,
                            pin_memory=True)

    print("Creating model")
    if args.model.lower() == 'mobilenet_v3_small':
        model = models.__dict__[args.model.lower()](pretrained=args.pretrained, width_mult=args.width, num_classes=len(dataset.classes))
    model.to(device)
    model_name = args.model.lower() + f'_{args.width}'

    opt_name = args.opt.lower()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum,
                                        weight_decay=args.weight_decay, eps=0.0316, alpha=0.9)
    else:
        raise RuntimeError("Invalid optimizer {}. Only SGD and RMSprop are supported.".format(args.opt))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    model_without_ddp = model
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    logger = TensorBoardLogger("lightning_logs", name=f"{model_name}-{args.dataset.lower()}", default_hp_metric=False)
    pl_model = LightningModel(model, learning_rate=args.lr, datamodule=dataset, optimizer=optimizer, lr_scheduler=lr_scheduler)

    checkpoint_callback = ModelCheckpoint(monitor='valid_acc',
                                          filename=f'{model_name}-{args.dataset.lower()}'+'-{epoch}-{valid_loss:.2f}-{valid_acc:.2f}',
                                          save_top_k=3)

    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, enable_progress_bar=False,
                         callbacks=[checkpoint_callback])
    trainer.fit(model=pl_model, datamodule=dataset, )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mobilenet_v3_small', help='model')
    parser.add_argument('--dataset', default='cifar10', help='dataset')
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