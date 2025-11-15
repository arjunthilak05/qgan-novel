"""
Entropy-Regularized GAN Loss

THE KEY INNOVATION OF THIS WORK

L_G = -log(D(G(z))) - α × S(ρ_A)
         ↑                    ↑
   Adversarial Loss    NOVEL CONTRIBUTION
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import numpy as np
from ..entropy import compute_entanglement_entropy, renyi_entanglement_entropy


class EntropyRegularizedLoss:
    """
    Entropy-Regularized GAN Loss.

    Generator loss:
        L_G = -log(D(G(z))) - α × S(ρ_A)

    where:
        - First term: standard adversarial loss
        - Second term: YOUR NOVEL CONTRIBUTION (entropy regularization)
        - α: regularization strength (hyperparameter)
        - S(ρ_A): entanglement entropy of subsystem A

    Args:
        alpha: Entropy regularization strength
        partition_size: Number of qubits in subsystem A
        entropy_type: Type of entropy ('von_neumann', 'renyi')
        renyi_alpha: Alpha parameter for Rényi entropy
        target_entropy: Optional target entropy value (if None, maximize entropy)
        base: Logarithm base for entropy calculation
    """

    def __init__(
        self,
        alpha: float = 0.1,
        partition_size: int = 2,
        entropy_type: str = "von_neumann",
        renyi_alpha: float = 2.0,
        target_entropy: Optional[float] = None,
        base: str = "2",
    ):
        self.alpha = alpha
        self.partition_size = partition_size
        self.entropy_type = entropy_type
        self.renyi_alpha = renyi_alpha
        self.target_entropy = target_entropy
        self.base = base

        # Track entropy history
        self.entropy_history = []

    def compute_entropy(self, state_vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute entanglement entropy for batch of quantum states.

        Args:
            state_vectors: Batch of quantum state vectors (batch_size, 2^n_qubits)

        Returns:
            Mean entropy across batch
        """
        if self.entropy_type == "von_neumann":
            # Compute entropy for each state in batch
            entropies = []
            for state in state_vectors:
                entropy = compute_entanglement_entropy(
                    state, self.partition_size, base=self.base
                )
                entropies.append(entropy)
            entropy = torch.stack(entropies).mean()

        elif self.entropy_type == "renyi":
            entropies = []
            for state in state_vectors:
                entropy = renyi_entanglement_entropy(
                    state, self.partition_size, alpha=self.renyi_alpha, base=self.base
                )
                entropies.append(entropy)
            entropy = torch.stack(entropies).mean()

        else:
            raise ValueError(f"Unknown entropy type: {self.entropy_type}")

        return entropy

    def discriminator_loss(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminator loss (same as vanilla GAN).

        Args:
            real_output: D(x_real)
            fake_output: D(G(z))

        Returns:
            Discriminator loss
        """
        real_loss = -torch.log(real_output + 1e-8).mean()
        fake_loss = -torch.log(1 - fake_output + 1e-8).mean()
        d_loss = real_loss + fake_loss

        return d_loss

    def generator_loss(
        self,
        fake_output: torch.Tensor,
        state_vectors: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generator loss with entropy regularization.

        THIS IS THE CORE OF THE NOVEL METHOD!

        L_G = -log(D(G(z))) - α × S(ρ_A)

        Args:
            fake_output: Discriminator output for fake data D(G(z))
            state_vectors: Quantum state vectors from generator

        Returns:
            (total_loss, loss_dict)

        where loss_dict contains:
            - 'adversarial': adversarial loss term
            - 'entropy': entropy regularization term
            - 'entropy_value': actual entropy value
            - 'total': total loss
        """
        # Adversarial loss: -log(D(G(z)))
        adversarial_loss = -torch.log(fake_output + 1e-8).mean()

        # Compute entanglement entropy
        entropy = self.compute_entropy(state_vectors)

        # Store entropy value (for logging)
        self.entropy_history.append(entropy.item())

        # Entropy regularization term
        if self.target_entropy is not None:
            # Penalty for deviating from target entropy
            entropy_term = -self.alpha * torch.abs(entropy - self.target_entropy)
        else:
            # Reward high entropy (encourages diversity)
            entropy_term = -self.alpha * entropy

        # Total generator loss
        total_loss = adversarial_loss + entropy_term

        # Prepare loss dict for logging
        loss_dict = {
            "adversarial": adversarial_loss.item(),
            "entropy_reg": entropy_term.item(),
            "entropy_value": entropy.item(),
            "total": total_loss.item(),
        }

        return total_loss, loss_dict

    def get_recent_entropy(self, n: int = 10) -> float:
        """Get mean entropy from last n iterations."""
        if len(self.entropy_history) < n:
            return np.mean(self.entropy_history) if self.entropy_history else 0.0
        return np.mean(self.entropy_history[-n:])

    def get_entropy_std(self, n: int = 10) -> float:
        """Get std of entropy from last n iterations."""
        if len(self.entropy_history) < n:
            return np.std(self.entropy_history) if self.entropy_history else 0.0
        return np.std(self.entropy_history[-n:])


class AdaptiveAlphaScheduler:
    """
    Adaptive scheduler for entropy regularization strength α.

    Adjusts α based on:
    - Entropy stability (decrease if entropy is stable)
    - Mode coverage (increase if mode collapse detected)

    Args:
        initial_alpha: Starting α value
        min_alpha: Minimum α value
        max_alpha: Maximum α value
        decay_rate: Decay rate when entropy is stable
        patience: Number of epochs before adjusting α
    """

    def __init__(
        self,
        initial_alpha: float = 0.5,
        min_alpha: float = 0.01,
        max_alpha: float = 1.0,
        decay_rate: float = 0.95,
        patience: int = 50,
    ):
        self.alpha = initial_alpha
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.decay_rate = decay_rate
        self.patience = patience

        self.entropy_history = []
        self.mode_coverage_history = []
        self.epochs_since_update = 0

    def step(
        self,
        entropy: float,
        mode_coverage: Optional[float] = None,
    ) -> float:
        """
        Update α based on entropy and mode coverage.

        Args:
            entropy: Current entropy value
            mode_coverage: Current mode coverage (if available)

        Returns:
            Updated α value
        """
        self.entropy_history.append(entropy)
        if mode_coverage is not None:
            self.mode_coverage_history.append(mode_coverage)

        self.epochs_since_update += 1

        # Only update after patience epochs
        if self.epochs_since_update < self.patience:
            return self.alpha

        # Check if entropy is stable
        if len(self.entropy_history) >= self.patience:
            recent_entropy = self.entropy_history[-self.patience :]
            entropy_mean = np.mean(recent_entropy)
            entropy_std = np.std(recent_entropy)

            # If entropy is stable and high, decrease α
            if entropy_std < 0.1 and entropy_mean > 1.0:
                self.alpha = max(self.min_alpha, self.alpha * self.decay_rate)
                self.epochs_since_update = 0

        # Check for mode collapse
        if mode_coverage is not None and len(self.mode_coverage_history) >= 10:
            recent_coverage = self.mode_coverage_history[-10:]
            mean_coverage = np.mean(recent_coverage)

            # If mode collapse detected, increase α
            if mean_coverage < 0.7:
                self.alpha = min(self.max_alpha, self.alpha * 1.1)
                self.epochs_since_update = 0

        return self.alpha

    def get_alpha(self) -> float:
        """Get current α value."""
        return self.alpha


def create_loss_function(config: dict) -> EntropyRegularizedLoss:
    """
    Factory function to create loss from config.

    Args:
        config: Configuration dictionary

    Returns:
        Loss function instance
    """
    loss_config = config["loss"]

    if loss_config["type"] == "vanilla_gan":
        # Baseline: no entropy regularization
        return EntropyRegularizedLoss(alpha=0.0)

    elif loss_config["type"] == "entropy_regularized":
        # Your method: with entropy regularization
        entropy_config = loss_config.get("entropy", {})

        return EntropyRegularizedLoss(
            alpha=loss_config["alpha"],
            partition_size=entropy_config.get("partition_size", 2),
            entropy_type=entropy_config.get("type", "von_neumann"),
            renyi_alpha=entropy_config.get("renyi_alpha", 2.0),
            target_entropy=entropy_config.get("target_entropy", None),
            base=entropy_config.get("base", "2"),
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")


__all__ = [
    "EntropyRegularizedLoss",
    "AdaptiveAlphaScheduler",
    "create_loss_function",
]
