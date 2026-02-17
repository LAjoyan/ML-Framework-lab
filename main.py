import argparse
import yaml

from src.dataset import get_dataloaders

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

    train_loader, test_loader = get_dataloaders(config)
    x, y = next(iter(train_loader))
    print("Train batch:", x.shape, y.shape)


if __name__ == "__main__":
    main()
