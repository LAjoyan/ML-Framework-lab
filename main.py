import argparse
import yaml

from src.dataset import get_dataloaders
from src.model import SimpleCnn
import torch


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    print("Loaded config:")
    print(config)

    train_loader, val_loader = get_dataloaders(config)

    x, y = next(iter(train_loader))
    print("Train batch:", x.shape, y.shape)

    xv, yv = next(iter(val_loader))
    print("Val batch:", xv.shape, yv.shape)

    model = SimpleCnn(num_classes=10)

    x, y = next(iter(train_loader))
    logits = model(x)

    print("Forward pass:")
    print("logits shape:", logits.shape)


if __name__ == "__main__":
    main()
