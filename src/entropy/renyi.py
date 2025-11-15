"""
Rényi Entropy

Alternative entropy measure that generalizes von Neumann entropy.

Rényi-α entropy: S_α(ρ) = 1/(1-α) log(Tr(ρ^α))

Special cases:
- α → 1: von Neumann entropy
- α = 2: Collision entropy (computationally cheaper!)
- α → ∞: Min-entropy
"""

import numpy as np
import torch
from typing import Union
from .von_neumann import compute_reduced_density_matrix


def renyi_entropy(
    rho: Union[np.ndarray, torch.Tensor],
    alpha: float = 2.0,
    base: str = "2",
    epsilon: float = 1e-12,
) -> Union[float, torch.Tensor]:
    """
    Compute Rényi entropy of order α.

    S_α(ρ) = 1/(1-α) log(Tr(ρ^α))

    Args:
        rho: Density matrix
        alpha: Rényi parameter (α > 0, α ≠ 1)
        base: Logarithm base
        epsilon: Small value for numerical stability

    Returns:
        Rényi entropy
    """
    assert alpha > 0 and alpha != 1, "Alpha must be positive and not equal to 1"

    is_torch = isinstance(rho, torch.Tensor)

    # Handle batched input
    if len(rho.shape) == 2:
        rho = rho.reshape(1, rho.shape[0], rho.shape[1])
        was_single = True
    else:
        was_single = False

    batch_size = rho.shape[0]
    entropies = []

    for i in range(batch_size):
        rho_i = rho[i]

        if is_torch:
            # Compute eigenvalues
            eigenvalues = torch.linalg.eigvalsh(rho_i)

            # Filter out numerical zeros
            eigenvalues = eigenvalues[eigenvalues > epsilon]

            # Compute Tr(ρ^α) = sum(λ^α)
            trace_rho_alpha = torch.sum(torch.pow(eigenvalues, alpha))

            # Compute Rényi entropy
            if base == "2":
                log_trace = torch.log2(trace_rho_alpha)
            elif base == "e":
                log_trace = torch.log(trace_rho_alpha)
            elif base == "10":
                log_trace = torch.log10(trace_rho_alpha)
            else:
                raise ValueError(f"Unknown base: {base}")

            entropy = (1.0 / (1.0 - alpha)) * log_trace
            entropies.append(entropy)

        else:  # numpy
            eigenvalues = np.linalg.eigvalsh(rho_i)
            eigenvalues = eigenvalues[eigenvalues > epsilon]

            trace_rho_alpha = np.sum(np.power(eigenvalues, alpha))

            if base == "2":
                log_trace = np.log2(trace_rho_alpha)
            elif base == "e":
                log_trace = np.log(trace_rho_alpha)
            elif base == "10":
                log_trace = np.log10(trace_rho_alpha)
            else:
                raise ValueError(f"Unknown base: {base}")

            entropy = (1.0 / (1.0 - alpha)) * log_trace
            entropies.append(entropy)

    if is_torch:
        entropies = torch.stack(entropies)
    else:
        entropies = np.array(entropies)

    if was_single:
        return entropies[0]
    else:
        return entropies


def renyi_entanglement_entropy(
    state_vector: Union[np.ndarray, torch.Tensor],
    partition_size: int,
    alpha: float = 2.0,
    base: str = "2",
) -> Union[float, torch.Tensor]:
    """
    Compute Rényi entanglement entropy.

    Useful alternative to von Neumann entropy:
    - α = 2 is computationally cheaper (no logarithm of eigenvalues)
    - Can be more robust to numerical noise

    Args:
        state_vector: Quantum state vector
        partition_size: Number of qubits in subsystem A
        alpha: Rényi parameter
        base: Logarithm base

    Returns:
        Rényi entanglement entropy
    """
    # Compute reduced density matrix
    rho_A = compute_reduced_density_matrix(state_vector, partition_size)

    # Compute Rényi entropy
    entropy = renyi_entropy(rho_A, alpha=alpha, base=base)

    return entropy


__all__ = [
    "renyi_entropy",
    "renyi_entanglement_entropy",
]
