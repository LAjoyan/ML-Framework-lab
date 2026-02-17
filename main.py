import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import SimpleCnn
from src.train import train_one_epoch, evaluate


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

    model = SimpleCnn(num_classes=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    epochs = int(config["epochs"])

    # Note:
    # We report training loss and validation loss/accuracy.
    # The Kaggle test set does not contain labels, so we cannot compute test loss.
    # Therefore, we split the training data into training/validation sets
    # and use the validation set to evaluate generalization during development.

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
