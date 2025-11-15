"""
Quantum Generator for ER-QGAN

Implements parametrized quantum circuits (PQCs) as the generator
with support for multiple ansatze and GPU acceleration.
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from typing import Optional, Callable, Tuple
from .circuit_ansatze import (
    hardware_efficient_ansatz,
    strongly_entangling_ansatz,
    simple_ansatz,
)


class QuantumGenerator(nn.Module):
    """
    Quantum Generator using PennyLane.

    Converts classical noise z -> quantum circuit -> classical data sample

    Args:
        n_qubits: Number of qubits in the circuit
        circuit_depth: Number of layers in the parametrized circuit
        output_dim: Dimension of output data (e.g., 2 for 2D distributions)
        ansatz: Circuit ansatz type ('hardware_efficient', 'strongly_entangling', 'simple')
        simulator: PennyLane device ('default.qubit', 'lightning.gpu', etc.)
        device: Torch device ('cuda' or 'cpu')
    """

    def __init__(
        self,
        n_qubits: int = 8,
        circuit_depth: int = 3,
        output_dim: int = 2,
        ansatz: str = "hardware_efficient",
        simulator: str = "lightning.gpu",
        device: str = "cuda:0",
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.output_dim = output_dim
        self.ansatz_type = ansatz
        self.device_name = device

        # Create quantum device
        if torch.cuda.is_available() and "gpu" in simulator.lower():
            self.qdevice = qml.device(simulator, wires=n_qubits)
        else:
            # Fallback to CPU
            self.qdevice = qml.device("default.qubit", wires=n_qubits)

        # Select ansatz function
        self.ansatz_fn = self._get_ansatz_function(ansatz)

        # Calculate number of parameters needed
        self.n_params = self._calculate_n_params()

        # Classical preprocessing: noise -> circuit parameters
        # This maps input noise to quantum circuit parameters
        self.noise_to_params = nn.Sequential(
            nn.Linear(self.n_qubits, 64),
            nn.Tanh(),
            nn.Linear(64, self.n_params),
        )

        # Classical postprocessing: measurement -> output data
        # This maps quantum measurements to output space
        self.measure_to_output = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )

        # Create quantum circuit
        self.qnode = qml.QNode(
            self._quantum_circuit,
            self.qdevice,
            interface="torch",
            diff_method="backprop",  # Efficient for simulators
        )

        # For entropy calculation - need separate qnode for state vector
        self.state_qnode = qml.QNode(
            self._quantum_circuit_state,
            self.qdevice,
            interface="torch",
            diff_method="backprop",
        )

    def _get_ansatz_function(self, ansatz: str) -> Callable:
        """Get ansatz function by name."""
        ansatz_map = {
            "hardware_efficient": hardware_efficient_ansatz,
            "strongly_entangling": strongly_entangling_ansatz,
            "simple": simple_ansatz,
        }

        if ansatz not in ansatz_map:
            raise ValueError(
                f"Unknown ansatz '{ansatz}'. Choose from {list(ansatz_map.keys())}"
            )

        return ansatz_map[ansatz]

    def _calculate_n_params(self) -> int:
        """Calculate number of parameters for the chosen ansatz."""
        if self.ansatz_type == "hardware_efficient":
            # 3 rotations per qubit per layer
            return self.circuit_depth * self.n_qubits * 3
        elif self.ansatz_type == "strongly_entangling":
            # 3 rotations per qubit per layer (PennyLane StronglyEntanglingLayers)
            return self.circuit_depth * self.n_qubits * 3
        elif self.ansatz_type == "simple":
            # 2 rotations per qubit per layer
            return self.circuit_depth * self.n_qubits * 2
        else:
            raise ValueError(f"Unknown ansatz: {self.ansatz_type}")

    def _quantum_circuit(self, params: torch.Tensor) -> torch.Tensor:
        """
        Quantum circuit that outputs measurements.

        Args:
            params: Circuit parameters (flattened)

        Returns:
            Measurement expectations for all qubits
        """
        # Reshape params for ansatz
        if self.ansatz_type in ["hardware_efficient", "simple"]:
            params_reshaped = params.reshape(self.circuit_depth, self.n_qubits, -1)
        else:  # strongly_entangling
            params_reshaped = params.reshape(self.circuit_depth, self.n_qubits, 3)

        # Apply ansatz
        self.ansatz_fn(params_reshaped, self.n_qubits, self.circuit_depth)

        # Measure all qubits in Z basis
        return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)])

    def _quantum_circuit_state(self, params: torch.Tensor) -> torch.Tensor:
        """
        Quantum circuit that outputs the full state vector.
        For entropy calculation.
        """
        if self.ansatz_type in ["hardware_efficient", "simple"]:
            params_reshaped = params.reshape(self.circuit_depth, self.n_qubits, -1)
        else:
            params_reshaped = params.reshape(self.circuit_depth, self.n_qubits, 3)

        self.ansatz_fn(params_reshaped, self.n_qubits, self.circuit_depth)

        return qml.state()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: noise -> quantum circuit -> output data

        Args:
            noise: Input noise tensor of shape (batch_size, noise_dim)

        Returns:
            Generated samples of shape (batch_size, output_dim)
        """
        batch_size = noise.shape[0]

        # Convert noise to circuit parameters
        circuit_params = self.noise_to_params(noise)  # (batch, n_params)

        # Run quantum circuit for each sample in batch
        measurements = []
        for i in range(batch_size):
            meas = self.qnode(circuit_params[i])
            measurements.append(meas)

        measurements = torch.stack(measurements)  # (batch, n_qubits)

        # Convert measurements to output space
        output = self.measure_to_output(measurements)  # (batch, output_dim)

        return output

    def get_quantum_state(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Get the quantum state vector for entropy calculation.

        Args:
            noise: Input noise tensor of shape (batch_size, noise_dim)

        Returns:
            State vectors of shape (batch_size, 2^n_qubits)
        """
        batch_size = noise.shape[0]

        # Convert noise to circuit parameters
        circuit_params = self.noise_to_params(noise)

        # Get state vectors
        states = []
        for i in range(batch_size):
            state = self.state_qnode(circuit_params[i])
            states.append(state)

        return torch.stack(states)  # (batch, 2^n_qubits)

    def get_n_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


def create_generator(config: dict, device: str = "cuda:0") -> QuantumGenerator:
    """
    Factory function to create generator from config.

    Args:
        config: Configuration dictionary
        device: Torch device

    Returns:
        QuantumGenerator instance
    """
    gen_config = config["model"]["generator"]
    data_config = config["data"]

    # Determine output dimension from dataset
    output_dim = 2  # Default for 2D distributions
    if "mnist" in data_config["name"].lower():
        output_dim = 28 * 28  # Flattened MNIST

    generator = QuantumGenerator(
        n_qubits=gen_config["n_qubits"],
        circuit_depth=gen_config["circuit_depth"],
        output_dim=output_dim,
        ansatz=gen_config["ansatz"],
        simulator=gen_config.get("simulator", "lightning.gpu"),
        device=device,
    )

    return generator.to(device)
