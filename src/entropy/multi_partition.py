"""
Multi-Partition Entanglement Entropy

Computes entropy across multiple bipartitions and returns average.
Can be more robust than single partition.
"""

import numpy as np
import torch
from typing import Union, List
from .von_neumann import compute_entanglement_entropy


def multi_partition_entropy(
    state_vector: Union[np.ndarray, torch.Tensor],
    partition_sizes: List[int],
    aggregation: str = "mean",
    base: str = "2",
) -> Union[float, torch.Tensor]:
    """
    Compute entropy across multiple bipartitions.

    Example: For 8 qubits, test partitions [1, 2, 3, 4]
    to get average entanglement across different cuts.

    Args:
        state_vector: Quantum state vector
        partition_sizes: List of partition sizes to test
        aggregation: How to combine entropies ('mean', 'max', 'min')
        base: Logarithm base

    Returns:
        Aggregated entropy value

    Example:
        >>> state = get_quantum_state(...)  # 8-qubit state
        >>> entropy = multi_partition_entropy(state, [1, 2, 3, 4])
        >>> # Returns average entropy across 4 different bipartitions
    """
    is_torch = isinstance(state_vector, torch.Tensor)

    # Infer number of qubits
    if is_torch:
        n_qubits = int(torch.log2(torch.tensor(state_vector.shape[-1])).item())
    else:
        n_qubits = int(np.log2(state_vector.shape[-1]))

    # Validate partition sizes
    for p_size in partition_sizes:
        assert 0 < p_size < n_qubits, f"Invalid partition size: {p_size}"

    # Compute entropy for each partition
    entropies = []
    for p_size in partition_sizes:
        entropy = compute_entanglement_entropy(
            state_vector, partition_size=p_size, base=base
        )
        entropies.append(entropy)

    # Aggregate
    if is_torch:
        entropies = torch.stack(entropies)
        if aggregation == "mean":
            result = torch.mean(entropies)
        elif aggregation == "max":
            result = torch.max(entropies)
        elif aggregation == "min":
            result = torch.min(entropies)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    else:
        entropies = np.array(entropies)
        if aggregation == "mean":
            result = np.mean(entropies)
        elif aggregation == "max":
            result = np.max(entropies)
        elif aggregation == "min":
            result = np.min(entropies)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    return result


def all_bipartitions_entropy(
    state_vector: Union[np.ndarray, torch.Tensor],
    base: str = "2",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute entropy for ALL possible bipartitions.

    For n qubits, computes S for partitions [1, 2, ..., n-1]

    Args:
        state_vector: Quantum state vector
        base: Logarithm base

    Returns:
        Array of entropies for each partition size
    """
    is_torch = isinstance(state_vector, torch.Tensor)

    # Infer number of qubits
    if is_torch:
        n_qubits = int(torch.log2(torch.tensor(state_vector.shape[-1])).item())
    else:
        n_qubits = int(np.log2(state_vector.shape[-1]))

    # Compute for all partitions
    partition_sizes = list(range(1, n_qubits))
    entropies = []

    for p_size in partition_sizes:
        entropy = compute_entanglement_entropy(
            state_vector, partition_size=p_size, base=base
        )
        entropies.append(entropy)

    if is_torch:
        return torch.stack(entropies)
    else:
        return np.array(entropies)


__all__ = [
    "multi_partition_entropy",
    "all_bipartitions_entropy",
]
