import pytorch_lightning as L

from src.dataset import get_dataloaders
from src.model import SimpleCnn


def run_experiment(config: dict):
    train_loader, val_loader, test_loader = get_dataloaders(config)

    model = SimpleCnn(num_classes=10, lr=config["learning_rate"])

    trainer = L.Trainer(
        max_epochs=config["epochs"],
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(model, train_loader, val_loader)

    test_results = trainer.test(model, test_loader)[0]

    return {"config": config, "test_acc": test_results["test_acc"]}


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
        print(f"\n>>> Starting {config['name']}...")
        result = run_experiment(config)
        results.append(result)

    print("\nFinal summary:")
    for r in results:
        print(f"{r['config']['name']} | test_acc={r['test_acc']:.4f}")


if __name__ == "__main__":
    main()
