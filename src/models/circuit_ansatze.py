"""
Quantum Circuit Ansatze

Different parametrized quantum circuit architectures for the generator.
"""

import pennylane as qml
import torch
from typing import Optional


def hardware_efficient_ansatz(
    params: torch.Tensor,
    n_qubits: int,
    circuit_depth: int,
) -> None:
    """
    Hardware-efficient ansatz with single-qubit rotations and CNOT entangling.

    Structure per layer:
    - RY, RZ, RY rotations on each qubit
    - Circular CNOT ladder for entanglement

    Args:
        params: Parameters of shape (circuit_depth, n_qubits, 3)
        n_qubits: Number of qubits
        circuit_depth: Number of layers
    """
    for layer in range(circuit_depth):
        # Single-qubit rotations
        for i in range(n_qubits):
            qml.RY(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)
            qml.RY(params[layer, i, 2], wires=i)

        # Entangling layer (circular CNOT ladder)
        if layer < circuit_depth - 1:  # No entangling after last layer
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])


def strongly_entangling_ansatz(
    params: torch.Tensor,
    n_qubits: int,
    circuit_depth: int,
) -> None:
    """
    PennyLane's StronglyEntanglingLayers template.

    Creates maximal entanglement between qubits.

    Args:
        params: Parameters of shape (circuit_depth, n_qubits, 3)
        n_qubits: Number of qubits
        circuit_depth: Number of layers
    """
    qml.StronglyEntanglingLayers(
        weights=params,
        wires=range(n_qubits),
    )


def simple_ansatz(
    params: torch.Tensor,
    n_qubits: int,
    circuit_depth: int,
) -> None:
    """
    Simple ansatz with RY rotations and CNOT ladder.

    Lighter than hardware-efficient, useful for testing.

    Args:
        params: Parameters of shape (circuit_depth, n_qubits, 2)
        n_qubits: Number of qubits
        circuit_depth: Number of layers
    """
    for layer in range(circuit_depth):
        # Single-qubit rotations
        for i in range(n_qubits):
            qml.RY(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)

        # Entangling layer
        if layer < circuit_depth - 1:
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])


def custom_ansatz(
    params: torch.Tensor,
    n_qubits: int,
    circuit_depth: int,
) -> None:
    """
    Custom ansatz - modify as needed for experiments.

    Args:
        params: Parameters of shape (circuit_depth, n_qubits, n_params_per_qubit)
        n_qubits: Number of qubits
        circuit_depth: Number of layers
    """
    for layer in range(circuit_depth):
        # Example: Hadamard + parametrized rotations
        for i in range(n_qubits):
            if layer == 0:
                qml.Hadamard(wires=i)
            qml.RY(params[layer, i, 0], wires=i)
            qml.RZ(params[layer, i, 1], wires=i)

        # All-to-all entanglement (expensive but maximal entanglement)
        if layer < circuit_depth - 1:
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    qml.CNOT(wires=[i, j])
