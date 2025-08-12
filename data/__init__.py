# from .class_dataset import ClassDataset
# from .pair_dataset import PairDataset
# from .item_dataset import ItemDataset
# from .ijb_dataset import IJBDataset
from .pytorch_dataset import MNIST, ImageNet, CIFAR10, CIFAR100, SVHN
#from . import *

__all__ = [
    'MNIST', 'ImageNet', 'CIFAR10', 'CIFAR100', 'SVHN',
]

DATASET_CLASSES = {
    'mnist': MNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'svhn': SVHN,
    'imagenet': ImageNet,
}

def get_dataset_class(name: str):
    try:
        return DATASET_CLASSES[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown dataset: {name}. Available options: {list(DATASET_CLASSES.keys())}")