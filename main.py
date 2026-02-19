import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import get_dataloaders
from src.model import SimpleCnn
from src.train import train_one_epoch, evaluate

def run_experiment(config: dict):
    print("\nRunning experiment with config:")
    print(config)

    train_loader, val_loader, test_loader = get_dataloaders(config)

    model = SimpleCnn(num_classes=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    epochs = int(config["epochs"])

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test | loss={test_loss:.4f} | acc={test_acc:.4f}")

    return {
        "config": config,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }

def main():
    experiments = [
        {
            "name": "exp1",
            "epochs": 3,
            "batch_size": 64,
            "learning_rate": 0.001,
            "val_split": 0.1,
            "seed": 42,
        },
        {
            "name": "exp2",
            "epochs": 3,
            "batch_size": 64,
            "learning_rate": 0.01,
            "val_split": 0.1,
            "seed": 42,
        },
        {
            "name": "exp3",
            "epochs": 3,
            "batch_size": 128,
            "learning_rate": 0.001,
            "val_split": 0.1,
            "seed": 42,
        },
    ]

    results = []

    for config in experiments:
        result = run_experiment(config)
        results.append(result)

    print("\nFinal summary:")
    for r in results:
        print(
            f"{r['config']['name']} | "
            f"test_acc={r['test_acc']:.4f}"
        )


if __name__ == "__main__":
    main()
