import argparse
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def evaluate_model(model_name, device_type):
    # Set device
    device = torch.device(device_type if torch.cuda.is_available() else "cpu")

    # Load model
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not supported")

    model.to(device)
    model.eval()

    # Load dataset
    from data.pytorch_dataset import CIFAR10

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

    # Evaluate model
    total_time = 0
    correct = 0
    total = 0

    trainer = pl.Trainer()
    trainer.fit(model=model, datamodule=dataset)



    print(f"Model: {model_name}")
    print(f"Device: {device_type}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Inference Time: {avg_inference_time:.6f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate computer vision models on embedded devices")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to evaluate (e.g., resnet18, mobilenet_v2)")
    parser.add_argument("--device", type=str, default="cuda", help="Device type (e.g., cuda, cpu)")

    args = parser.parse_args()

    evaluate_model(args.model, args.device)
