"""
KL Divergence Metric

Measure distributional similarity between real and generated data.
"""

import numpy as np
import torch
from typing import Union
from scipy.stats import entropy


def compute_kl_divergence(
    real_samples: Union[np.ndarray, torch.Tensor],
    generated_samples: Union[np.ndarray, torch.Tensor],
    bins: int = 50,
) -> float:
    """
    Estimate KL divergence between real and generated distributions.

    KL(P||Q) = sum P(x) log(P(x)/Q(x))

    Lower is better (0 = perfect match).

    Args:
        real_samples: Real data samples
        generated_samples: Generated data samples
        bins: Number of histogram bins

    Returns:
        KL divergence estimate
    """
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.detach().cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.detach().cpu().numpy()

    # For multidimensional data, compute KL per dimension and average
    if len(real_samples.shape) > 1:
        kl_per_dim = []
        for dim in range(real_samples.shape[1]):
            kl = compute_kl_divergence_1d(
                real_samples[:, dim],
                generated_samples[:, dim],
                bins=bins,
            )
            kl_per_dim.append(kl)
        return np.mean(kl_per_dim)
    else:
        return compute_kl_divergence_1d(real_samples, generated_samples, bins=bins)


def compute_kl_divergence_1d(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    bins: int = 50,
) -> float:
    """Compute KL divergence for 1D data."""
    # Create histograms
    hist_range = (
        min(real_samples.min(), generated_samples.min()),
        max(real_samples.max(), generated_samples.max()),
    )

    real_hist, _ = np.histogram(real_samples, bins=bins, range=hist_range, density=True)
    gen_hist, _ = np.histogram(generated_samples, bins=bins, range=hist_range, density=True)

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    real_hist = real_hist + epsilon
    gen_hist = gen_hist + epsilon

    # Normalize
    real_hist = real_hist / real_hist.sum()
    gen_hist = gen_hist / gen_hist.sum()

    # Compute KL divergence
    kl = entropy(real_hist, gen_hist)

    return float(kl)


__all__ = [
    "compute_kl_divergence",
    "compute_kl_divergence_1d",
]
