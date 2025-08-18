import argparse

import pytorch_lightning as pl
import torch
import torchvision.models as models
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from data import get_dataset_class
from model.LightningModel import LightningModel, LightningModelLight

from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion
from neural_compressor.quantization import fit


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    torch.backends.cudnn.benchmark = True

    print('############### Initializing Dataset ###############')
    transform_preset = 'default' if not 'efficientnet' in args.model.lower() else args.model.lower()
    dataset = get_dataset_class(args.dataset)(data_dir=args.data_dir, train_val_split=0.2, transform_preset=transform_preset, classes=1000)

    dataset.setup('fit')
    dataset.train_dataloader(batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=16,
                             drop_last=True,
                             pin_memory=True)
    dataset.val_dataloader(batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=16,
                           pin_memory=True)
    
    dataset.test_dataloader(batch_size=args.batch_size,
                            num_workers=16,
                            pin_memory=True)




    print('############### Creating model ###############')
    model = models.__dict__[args.model.lower()](pretrained=args.pretrained)
    model.to(device)
    model_name = args.model.lower() + f'_{args.width}'


    logger = TensorBoardLogger("lightning_logs", name=f"{model_name}-{args.dataset.lower()}", default_hp_metric=False)
    pl_model = LightningModelLight(model, datamodule=dataset, verbose=args.verbose)

    checkpoint_callback = ModelCheckpoint(monitor='valid_loss',
                                          filename=f'{model_name}-{args.dataset.lower()}'+'-{epoch}-{valid_loss:.2f}-{valid_acc:.2f}',
                                          save_top_k=3)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(accelerator='gpu' if args.device == 'cuda' else 'cpu', logger=logger, max_epochs=args.epochs, enable_progress_bar=False,
                         callbacks=[checkpoint_callback, lr_monitor])
    




    print('############### Testing FP32 model ###############')
    fp32_acc = trainer.test(model=pl_model, datamodule=dataset, )
    print('Results FP32 evaluation: ', fp32_acc)

    def evaluate(model):
        accuracy = trainer.test(model=pl_model.model, datamodule=dataset, )
        return accuracy
    
    print('############### Quantizing model ###############')
    from neural_compressor import PostTrainingQuantConfig
    from neural_compressor import quantization

    if 'efficientnet' in args.model.lower():
        # To reduce tuning time and get the result faster, the efficient net series model use the MSE_V2 strategy by default.
        from neural_compressor.config import TuningCriterion
        tuning_criterion = TuningCriterion(strategy="mse_v2")
        conf = PostTrainingQuantConfig(quant_level=1, tuning_criterion=tuning_criterion)
        from neural_compressor.metric import METRICS
        metrics = METRICS('pytorch')
        top1 = metrics['topk']()
        q_model = quantization.fit(pl_model.model,
                                    conf,
                                    calib_dataloader=dataset.val_dataloader,
                                    eval_dataloader=dataset.val_dataloader,
                                    eval_metric=top1)
    
    accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
    tuning_criterion = TuningCriterion(max_trials=600)
    conf = PostTrainingQuantConfig(
        approach="static", backend="default", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion
    )
    q_model = fit(model=pl_model.model, conf=conf, calib_dataloader=dataset, eval_func=evaluate)
    q_model.save("./model/quantized/")




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
    parser.add_argument('--verbose', default=0, type=int, help='how much information should be given')

    parser.add_argument("-w",
                        "--width",
                        dest="width",
                        type=float,
                        help="Select width multiplier of model",
                        default=1.0)

    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=True,
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    arguments = parser.parse_args()
    print(f'arguments: {arguments}')
    main(arguments)