"""
Von Neumann Entanglement Entropy Calculation

THE CORE INNOVATION OF THIS WORK

Computes S(ρ_A) = -Tr(ρ_A log ρ_A) for quantum states,
which measures entanglement and is used as a regularization term.
"""

import numpy as np
import torch
from typing import Union, Tuple
import warnings


def compute_reduced_density_matrix(
    state_vector: Union[np.ndarray, torch.Tensor],
    partition_size: int,
    n_qubits: Optional[int] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute reduced density matrix by tracing out subsystem B.

    For a bipartite system A⊗B, computes ρ_A = Tr_B(|ψ⟩⟨ψ|)

    Args:
        state_vector: Quantum state vector of length 2^n
        partition_size: Number of qubits in subsystem A
        n_qubits: Total number of qubits (inferred if None)

    Returns:
        Reduced density matrix ρ_A of shape (2^partition_size, 2^partition_size)
    """
    # Convert to numpy if torch tensor
    is_torch = isinstance(state_vector, torch.Tensor)
    if is_torch:
        device = state_vector.device
        state_np = state_vector.detach().cpu().numpy()
    else:
        state_np = state_vector

    # Handle batched input
    if len(state_np.shape) == 1:
        state_np = state_np.reshape(1, -1)
        was_single = True
    else:
        was_single = False

    batch_size = state_np.shape[0]

    # Infer number of qubits
    if n_qubits is None:
        n_qubits = int(np.log2(state_np.shape[1]))

    # Validate
    assert state_np.shape[1] == 2**n_qubits, "State vector size must be 2^n_qubits"
    assert partition_size < n_qubits, "Partition must be smaller than total qubits"

    dim_A = 2**partition_size
    dim_B = 2 ** (n_qubits - partition_size)

    # Compute reduced density matrices for batch
    rho_A_batch = []

    for i in range(batch_size):
        psi = state_np[i]

        # Reshape state to matrix form: (dim_A, dim_B)
        psi_matrix = psi.reshape(dim_A, dim_B)

        # Compute reduced density matrix: ρ_A = ψ ψ^†
        # This is equivalent to Tr_B(|ψ⟩⟨ψ|)
        rho_A = psi_matrix @ psi_matrix.conj().T

        rho_A_batch.append(rho_A)

    rho_A_batch = np.stack(rho_A_batch)

    if was_single:
        rho_A_batch = rho_A_batch[0]

    # Convert back to torch if needed
    if is_torch:
        rho_A_batch = torch.from_numpy(rho_A_batch).to(device)

    return rho_A_batch


def von_neumann_entropy(
    rho: Union[np.ndarray, torch.Tensor],
    base: str = "2",
    epsilon: float = 1e-12,
) -> Union[float, torch.Tensor]:
    """
    Compute von Neumann entropy: S(ρ) = -Tr(ρ log ρ)

    Args:
        rho: Density matrix
        base: Logarithm base ('2', 'e', or '10')
        epsilon: Small value to avoid log(0)

    Returns:
        Von Neumann entropy (scalar or tensor)
    """
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

            # Filter out numerical zeros and negative values
            eigenvalues = eigenvalues[eigenvalues > epsilon]

            # Compute entropy
            if base == "2":
                log_eig = torch.log2(eigenvalues)
            elif base == "e":
                log_eig = torch.log(eigenvalues)
            elif base == "10":
                log_eig = torch.log10(eigenvalues)
            else:
                raise ValueError(f"Unknown base: {base}")

            entropy = -torch.sum(eigenvalues * log_eig)
            entropies.append(entropy)

        else:  # numpy
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(rho_i)

            # Filter out numerical zeros
            eigenvalues = eigenvalues[eigenvalues > epsilon]

            # Compute entropy
            if base == "2":
                log_eig = np.log2(eigenvalues)
            elif base == "e":
                log_eig = np.log(eigenvalues)
            elif base == "10":
                log_eig = np.log10(eigenvalues)
            else:
                raise ValueError(f"Unknown base: {base}")

            entropy = -np.sum(eigenvalues * log_eig)
            entropies.append(entropy)

    if is_torch:
        entropies = torch.stack(entropies)
    else:
        entropies = np.array(entropies)

    if was_single:
        return entropies[0]
    else:
        return entropies


def compute_entanglement_entropy(
    state_vector: Union[np.ndarray, torch.Tensor],
    partition_size: int,
    base: str = "2",
    epsilon: float = 1e-12,
) -> Union[float, torch.Tensor]:
    """
    Compute entanglement entropy for a bipartite system.

    THIS IS THE KEY FUNCTION USED IN THE LOSS!

    S(ρ_A) = -Tr(ρ_A log ρ_A)

    where ρ_A = Tr_B(|ψ⟩⟨ψ|) is the reduced density matrix.

    Args:
        state_vector: Quantum state vector
        partition_size: Number of qubits in subsystem A
        base: Logarithm base
        epsilon: Small value to avoid numerical issues

    Returns:
        Entanglement entropy (scalar or tensor for batch)

    Example:
        >>> state = torch.randn(16, dtype=torch.complex64)  # 4-qubit state
        >>> state = state / torch.norm(state)
        >>> entropy = compute_entanglement_entropy(state, partition_size=2)
        >>> print(f"Entanglement entropy: {entropy:.3f}")
    """
    # Compute reduced density matrix
    rho_A = compute_reduced_density_matrix(state_vector, partition_size)

    # Compute von Neumann entropy
    entropy = von_neumann_entropy(rho_A, base=base, epsilon=epsilon)

    return entropy


def batch_entanglement_entropy(
    state_vectors: torch.Tensor,
    partition_size: int,
    base: str = "2",
) -> torch.Tensor:
    """
    Compute entanglement entropy for a batch of states.

    More efficient than calling compute_entanglement_entropy in a loop.

    Args:
        state_vectors: Batch of state vectors (batch_size, 2^n_qubits)
        partition_size: Number of qubits in subsystem A
        base: Logarithm base

    Returns:
        Entropies of shape (batch_size,)
    """
    batch_size = state_vectors.shape[0]

    # Compute reduced density matrices for all states
    rho_A_batch = compute_reduced_density_matrix(state_vectors, partition_size)

    # Compute entropies
    entropies = von_neumann_entropy(rho_A_batch, base=base)

    return entropies


# Type hint helper
from typing import Optional

__all__ = [
    "compute_reduced_density_matrix",
    "von_neumann_entropy",
    "compute_entanglement_entropy",
    "batch_entanglement_entropy",
]
