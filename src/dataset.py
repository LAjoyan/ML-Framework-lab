import torch
from torch.utils.data import  DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(config: dict):
    batch_size = int(config["batch_size"])
    seed = int(config.get("seed", 42))
    val_split = float(config.get("val_split", 0.1))
    data_root = config.get("data_root", "data")
    num_workers = int(config.get("num_workers", 0))


    transform = transforms.ToTensor()

    full_train_ds = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=transform,
    )

    test_ds = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=transform,
    )

    n_total = len(full_train_ds)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train_ds, [n_train, n_val], generator=gen)


    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


    return train_loader, val_loader, test_loader
