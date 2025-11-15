"""
Synthetic Datasets for Mode Coverage Evaluation

Includes:
- 8-Gaussian mixture (2D) - standard test for mode collapse
- 25-Gaussian grid (2D) - harder test
- Rings dataset
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List


class GaussianMixture(Dataset):
    """
    Mixture of Gaussians dataset.

    Useful for testing mode coverage: can the GAN capture all modes?

    Args:
        n_samples: Number of samples
        n_modes: Number of Gaussian modes
        std: Standard deviation of each Gaussian
        radius: Radius for arranging modes in a circle
        grid: If True, arrange in grid instead of circle
    """

    def __init__(
        self,
        n_samples: int = 10000,
        n_modes: int = 8,
        std: float = 0.1,
        radius: float = 2.0,
        grid: bool = False,
    ):
        self.n_samples = n_samples
        self.n_modes = n_modes
        self.std = std
        self.radius = radius

        # Generate mode centers
        if grid:
            # Grid arrangement (for 25-Gaussian)
            grid_size = int(np.sqrt(n_modes))
            assert grid_size ** 2 == n_modes, "n_modes must be perfect square for grid"

            x = np.linspace(-radius, radius, grid_size)
            y = np.linspace(-radius, radius, grid_size)
            xx, yy = np.meshgrid(x, y)
            self.mode_centers = np.stack([xx.flatten(), yy.flatten()], axis=1)

        else:
            # Circular arrangement (for 8-Gaussian)
            angles = np.linspace(0, 2 * np.pi, n_modes, endpoint=False)
            self.mode_centers = radius * np.stack(
                [np.cos(angles), np.sin(angles)], axis=1
            )

        # Generate samples
        self.data = self._generate_samples()

    def _generate_samples(self) -> np.ndarray:
        """Generate samples from mixture of Gaussians."""
        samples = []

        # Sample from each mode equally
        samples_per_mode = self.n_samples // self.n_modes

        for center in self.mode_centers:
            mode_samples = np.random.randn(samples_per_mode, 2) * self.std + center
            samples.append(mode_samples)

        # Add remaining samples
        remaining = self.n_samples - samples_per_mode * self.n_modes
        if remaining > 0:
            random_modes = np.random.choice(self.n_modes, remaining)
            for mode_idx in random_modes:
                center = self.mode_centers[mode_idx]
                sample = np.random.randn(2) * self.std + center
                samples.append(sample.reshape(1, -1))

        samples = np.concatenate(samples, axis=0)

        # Shuffle
        np.random.shuffle(samples)

        return samples.astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])

    def get_mode_centers(self) -> np.ndarray:
        """Get the centers of all modes."""
        return self.mode_centers


def create_8gaussian(
    n_samples: int = 10000,
    std: float = 0.1,
    radius: float = 2.0,
) -> GaussianMixture:
    """
    Create 8-Gaussian dataset (standard mode collapse test).

    8 Gaussians arranged in a circle.

    Args:
        n_samples: Number of samples
        std: Standard deviation
        radius: Radius of circle

    Returns:
        GaussianMixture dataset
    """
    return GaussianMixture(
        n_samples=n_samples,
        n_modes=8,
        std=std,
        radius=radius,
        grid=False,
    )


def create_25gaussian(
    n_samples: int = 10000,
    std: float = 0.1,
    radius: float = 2.0,
) -> GaussianMixture:
    """
    Create 25-Gaussian grid dataset (harder test).

    25 Gaussians arranged in 5x5 grid.

    Args:
        n_samples: Number of samples
        std: Standard deviation
        radius: Grid extent

    Returns:
        GaussianMixture dataset
    """
    return GaussianMixture(
        n_samples=n_samples,
        n_modes=25,
        std=std,
        radius=radius,
        grid=True,
    )


class RingsDataset(Dataset):
    """
    Concentric rings dataset.

    Args:
        n_samples: Number of samples
        n_rings: Number of concentric rings
        radius_min: Minimum radius
        radius_max: Maximum radius
        std: Noise standard deviation
    """

    def __init__(
        self,
        n_samples: int = 10000,
        n_rings: int = 3,
        radius_min: float = 0.5,
        radius_max: float = 2.0,
        std: float = 0.05,
    ):
        self.n_samples = n_samples
        self.n_rings = n_rings

        # Ring radii
        self.radii = np.linspace(radius_min, radius_max, n_rings)

        # Generate samples
        self.data = self._generate_samples(std)

    def _generate_samples(self, std: float) -> np.ndarray:
        """Generate samples on concentric rings."""
        samples = []
        samples_per_ring = self.n_samples // self.n_rings

        for radius in self.radii:
            # Uniform angles
            angles = np.random.uniform(0, 2 * np.pi, samples_per_ring)

            # Points on ring + noise
            x = radius * np.cos(angles) + np.random.randn(samples_per_ring) * std
            y = radius * np.sin(angles) + np.random.randn(samples_per_ring) * std

            ring_samples = np.stack([x, y], axis=1)
            samples.append(ring_samples)

        samples = np.concatenate(samples, axis=0)
        np.random.shuffle(samples)

        return samples.astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.data[idx])


def create_dataloader(
    dataset_name: str,
    batch_size: int = 256,
    n_samples: int = 10000,
    **kwargs,
) -> DataLoader:
    """
    Factory function to create dataloader.

    Args:
        dataset_name: '8gaussian', '25gaussian', 'rings'
        batch_size: Batch size
        n_samples: Number of samples
        **kwargs: Additional dataset-specific arguments

    Returns:
        DataLoader
    """
    if dataset_name == "8gaussian":
        dataset = create_8gaussian(n_samples=n_samples, **kwargs)
    elif dataset_name == "25gaussian":
        dataset = create_25gaussian(n_samples=n_samples, **kwargs)
    elif dataset_name == "rings":
        dataset = RingsDataset(n_samples=n_samples, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


__all__ = [
    "GaussianMixture",
    "create_8gaussian",
    "create_25gaussian",
    "RingsDataset",
    "create_dataloader",
]
