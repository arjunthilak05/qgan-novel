"""
Diversity Metrics

Measure how diverse the generated samples are.
"""

import numpy as np
import torch
from typing import Union
from scipy.spatial.distance import pdist, squareform


def pairwise_distances(
    samples: Union[np.ndarray, torch.Tensor],
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute pairwise distances between samples.

    Args:
        samples: Data samples (n_samples, dim)
        metric: Distance metric

    Returns:
        Pairwise distance matrix (n_samples, n_samples)
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()

    distances = squareform(pdist(samples, metric=metric))
    return distances


def compute_diversity_score(
    samples: Union[np.ndarray, torch.Tensor],
    metric: str = "euclidean",
) -> float:
    """
    Compute diversity score as mean pairwise distance.

    Higher diversity = more spread out samples = better!

    Args:
        samples: Generated samples (n_samples, dim)
        metric: Distance metric

    Returns:
        Mean pairwise distance

    Example:
        >>> fake_data = generator(noise)  # (5000, 2)
        >>> diversity = compute_diversity_score(fake_data)
        >>> print(f"Diversity: {diversity:.3f}")
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()

    if len(samples) < 2:
        return 0.0

    distances = pdist(samples, metric=metric)
    return float(np.mean(distances))


__all__ = [
    "pairwise_distances",
    "compute_diversity_score",
]
