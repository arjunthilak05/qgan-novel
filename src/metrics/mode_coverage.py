"""
Mode Coverage Metric

THE PRIMARY EVALUATION METRIC

Measures what percentage of true distribution modes are covered
by generated samples. High mode coverage = no mode collapse!
"""

import numpy as np
import torch
from typing import Union, List
from scipy.spatial.distance import cdist


def compute_mode_coverage(
    generated_samples: Union[np.ndarray, torch.Tensor],
    mode_centers: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.3,
) -> float:
    """
    Compute mode coverage: percentage of modes covered by generated samples.

    A mode is "covered" if at least one generated sample is within threshold
    distance from the mode center.

    Args:
        generated_samples: Generated data points (n_samples, dim)
        mode_centers: True mode centers (n_modes, dim)
        threshold: Distance threshold for mode detection

    Returns:
        Mode coverage percentage (0.0 to 1.0)

    Example:
        >>> # 8-Gaussian dataset
        >>> mode_centers = dataset.get_mode_centers()  # (8, 2)
        >>> fake_data = generator(noise)  # (5000, 2)
        >>> coverage = compute_mode_coverage(fake_data, mode_centers, threshold=0.3)
        >>> print(f"Mode coverage: {coverage*100:.1f}%")
        Mode coverage: 87.5%  # Covered 7 out of 8 modes
    """
    # Convert to numpy
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.detach().cpu().numpy()
    if isinstance(mode_centers, torch.Tensor):
        mode_centers = mode_centers.detach().cpu().numpy()

    n_modes = len(mode_centers)

    # Compute distances from each sample to each mode
    distances = cdist(generated_samples, mode_centers)  # (n_samples, n_modes)

    # For each mode, check if any sample is within threshold
    min_distances = distances.min(axis=0)  # (n_modes,)
    covered_modes = (min_distances < threshold).sum()

    coverage = covered_modes / n_modes

    return coverage


class ModeCounter:
    """
    Track which modes are covered and count samples per mode.

    Useful for detailed analysis of mode collapse.

    Args:
        mode_centers: Centers of true distribution modes
        threshold: Distance threshold
    """

    def __init__(
        self,
        mode_centers: np.ndarray,
        threshold: float = 0.3,
    ):
        self.mode_centers = mode_centers
        self.threshold = threshold
        self.n_modes = len(mode_centers)

        self.mode_counts = np.zeros(self.n_modes, dtype=int)
        self.covered_modes = set()

    def update(self, generated_samples: Union[np.ndarray, torch.Tensor]):
        """
        Update mode counts with new generated samples.

        Args:
            generated_samples: Generated data points
        """
        if isinstance(generated_samples, torch.Tensor):
            generated_samples = generated_samples.detach().cpu().numpy()

        # Compute distances
        distances = cdist(generated_samples, self.mode_centers)

        # Assign each sample to nearest mode (if within threshold)
        nearest_modes = distances.argmin(axis=1)  # (n_samples,)
        min_distances = distances.min(axis=1)  # (n_samples,)

        # Count samples per mode
        for sample_idx, (mode_idx, dist) in enumerate(zip(nearest_modes, min_distances)):
            if dist < self.threshold:
                self.mode_counts[mode_idx] += 1
                self.covered_modes.add(mode_idx)

    def get_coverage(self) -> float:
        """Get current mode coverage."""
        return len(self.covered_modes) / self.n_modes

    def get_mode_counts(self) -> np.ndarray:
        """Get number of samples assigned to each mode."""
        return self.mode_counts

    def get_covered_modes(self) -> List[int]:
        """Get list of covered mode indices."""
        return sorted(list(self.covered_modes))

    def get_missing_modes(self) -> List[int]:
        """Get list of uncovered mode indices."""
        all_modes = set(range(self.n_modes))
        missing = all_modes - self.covered_modes
        return sorted(list(missing))

    def reset(self):
        """Reset all counts."""
        self.mode_counts = np.zeros(self.n_modes, dtype=int)
        self.covered_modes = set()

    def summary(self) -> dict:
        """Get summary statistics."""
        return {
            "coverage": self.get_coverage(),
            "covered_modes": self.get_covered_modes(),
            "missing_modes": self.get_missing_modes(),
            "samples_per_mode": self.mode_counts.tolist(),
            "min_samples": self.mode_counts.min(),
            "max_samples": self.mode_counts.max(),
            "mean_samples": self.mode_counts.mean(),
            "std_samples": self.mode_counts.std(),
        }


def compute_mode_quality(
    generated_samples: np.ndarray,
    mode_centers: np.ndarray,
    threshold: float = 0.3,
) -> dict:
    """
    Compute comprehensive mode quality metrics.

    Returns:
        Dictionary with:
        - coverage: mode coverage percentage
        - quality: average quality of covered modes
        - balance: evenness of sample distribution across modes
    """
    counter = ModeCounter(mode_centers, threshold)
    counter.update(generated_samples)

    coverage = counter.get_coverage()
    mode_counts = counter.get_mode_counts()

    # Quality: how well are covered modes represented?
    covered_counts = mode_counts[mode_counts > 0]
    if len(covered_counts) > 0:
        quality = covered_counts.mean()
    else:
        quality = 0.0

    # Balance: evenness of distribution (using entropy)
    if mode_counts.sum() > 0:
        probs = mode_counts / mode_counts.sum()
        probs = probs[probs > 0]  # Remove zeros
        balance = -np.sum(probs * np.log(probs + 1e-8))
        balance /= np.log(len(mode_centers))  # Normalize to [0, 1]
    else:
        balance = 0.0

    return {
        "coverage": coverage,
        "quality": quality,
        "balance": balance,
        "summary": counter.summary(),
    }


__all__ = [
    "compute_mode_coverage",
    "ModeCounter",
    "compute_mode_quality",
]
