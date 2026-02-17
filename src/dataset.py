import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def load_cifar_binary_images(path: str, num_images: int) -> np.ndarray:
    """
    Load raw uint8 CIFAR images stored as a flat binary file.
    Returns shape: (N, 32, 32, 3) in HWC.
    """
    expected_bytes = num_images * 32 * 32 * 3
    actual_bytes = os.path.getsize(path)

    if actual_bytes < expected_bytes:
        raise ValueError(
            f"{path} is too small. Expected at least {expected_bytes} bytes, got {actual_bytes}."
        )

    flat = np.fromfile(path, dtype=np.uint8, count=expected_bytes)
    images = flat.reshape(num_images, 32, 32, 3)  # HWC
    return images


class KaggleCifar10Dataset(Dataset):
    def __init__(self, images_path: str, labels_csv_path: str | None = None, num_images: int | None = None):
        if num_images is None:
            raise ValueError("num_images must be provided (e.g., 50000 for train, 300000 for test).")

        self.images = load_cifar_binary_images(images_path, num_images=num_images)
        self.labels = None

        if labels_csv_path is not None:
            df = pd.read_csv(labels_csv_path)  # columns: id,label
            label_names = df["label"].astype(str).tolist()

            classes = sorted(set(label_names))
            self.class_to_idx = {name: i for i, name in enumerate(classes)}
            self.labels = np.array([self.class_to_idx[name] for name in label_names], dtype=np.int64)

            if len(self.labels) != len(self.images):
                raise ValueError(f"Mismatch: {len(self.images)} images but {len(self.labels)} labels.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        x = self.images[idx]  # (32, 32, 3) uint8
        x = torch.tensor(x, dtype=torch.float32) / 255.0
        x = x.permute(2, 0, 1)  # CHW

        if self.labels is None:
            return x

        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def get_dataloaders(config: dict):
    batch_size = int(config["batch_size"])

    train_ds = KaggleCifar10Dataset(
        images_path="data/raw/train.npy",
        labels_csv_path="data/raw/trainLabels.csv",
        num_images=50000,
    )

    test_ds = KaggleCifar10Dataset(
        images_path="data/raw/test.npy",
        labels_csv_path=None,
        num_images=300000,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
