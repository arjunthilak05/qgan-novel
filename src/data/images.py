"""
Image dataset loaders (MNIST, Fashion-MNIST)
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple


def load_mnist(
    batch_size: int = 256,
    data_dir: str = "./data",
    subset_classes: list = None,
    flatten: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset.

    Args:
        batch_size: Batch size
        data_dir: Directory to save/load data
        subset_classes: If provided, only load these classes (e.g., [0, 1] for digits 0 and 1)
        flatten: If True, flatten images to 784-dim vectors

    Returns:
        (train_loader, test_loader)
    """
    if flatten:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.view(-1)),  # Flatten
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    # Subset if requested
    if subset_classes is not None:
        train_indices = [
            i for i, (_, label) in enumerate(train_dataset) if label in subset_classes
        ]
        test_indices = [
            i for i, (_, label) in enumerate(test_dataset) if label in subset_classes
        ]

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    return train_loader, test_loader


def load_fashion_mnist(
    batch_size: int = 256,
    data_dir: str = "./data",
    flatten: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load Fashion-MNIST dataset.

    Args:
        batch_size: Batch size
        data_dir: Directory to save/load data
        flatten: If True, flatten images

    Returns:
        (train_loader, test_loader)
    """
    if flatten:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    return train_loader, test_loader


__all__ = [
    "load_mnist",
    "load_fashion_mnist",
]
