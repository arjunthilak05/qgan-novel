"""
QGAN Trainer

Complete training loop for both baseline and entropy-regularized QGANs.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm
import os

from ..models import QuantumGenerator, Discriminator
from ..losses import EntropyRegularizedLoss
from ..metrics import compute_mode_coverage, compute_diversity_score, compute_kl_divergence


class QGANTrainer:
    """
    Trainer for Quantum GANs.

    Handles:
    - Training loop
    - Evaluation
    - Checkpointing
    - Logging (WandB/TensorBoard)

    Args:
        generator: Quantum generator
        discriminator: Classical discriminator
        loss_fn: Loss function
        device: Torch device
        config: Configuration dictionary
    """

    def __init__(
        self,
        generator: QuantumGenerator,
        discriminator: nn.Module,
        loss_fn: EntropyRegularizedLoss,
        device: str = "cuda:0",
        config: Optional[Dict] = None,
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.config = config or {}

        # Optimizers
        gen_config = config.get("training", {}).get("optimizer", {}).get("generator", {})
        disc_config = config.get("training", {}).get("optimizer", {}).get("discriminator", {})

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=gen_config.get("lr", 0.01),
            betas=gen_config.get("betas", [0.9, 0.999]),
        )

        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=disc_config.get("lr", 0.001),
            betas=disc_config.get("betas", [0.9, 0.999]),
        )

        # Training settings
        train_config = config.get("training", {})
        self.d_updates_per_g = train_config.get("d_updates_per_g", 5)
        self.gradient_clip = train_config.get("gradient_clip", None)

        # Logging
        self.use_wandb = config.get("logging", {}).get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("wandb not installed, disabling wandb logging")
                self.use_wandb = False

        # Checkpointing
        self.save_dir = config.get("logging", {}).get("save_dir", "results/logs")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/checkpoints", exist_ok=True)

        # Metrics history
        self.history = {
            "g_loss": [],
            "d_loss": [],
            "entropy": [],
            "mode_coverage": [],
            "diversity": [],
        }

    def train_step(
        self,
        real_data: torch.Tensor,
        noise_dim: int = 4,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            real_data: Real data batch
            noise_dim: Noise dimension

        Returns:
            Dictionary of losses
        """
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)

        metrics = {}

        # =============================
        # Train Discriminator
        # =============================
        for _ in range(self.d_updates_per_g):
            self.d_optimizer.zero_grad()

            # Generate fake data
            noise = torch.randn(batch_size, noise_dim, device=self.device)
            fake_data = self.generator(noise)

            # Discriminator outputs
            real_output = self.discriminator(real_data)
            fake_output = self.discriminator(fake_data.detach())

            # Discriminator loss
            d_loss = self.loss_fn.discriminator_loss(real_output, fake_output)

            # Backward
            d_loss.backward()

            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), self.gradient_clip
                )

            self.d_optimizer.step()

        metrics["d_loss"] = d_loss.item()

        # =============================
        # Train Generator
        # =============================
        self.g_optimizer.zero_grad()

        # Generate fake data
        noise = torch.randn(batch_size, noise_dim, device=self.device)
        fake_data = self.generator(noise)

        # Get quantum states for entropy calculation
        state_vectors = self.generator.get_quantum_state(noise)

        # Discriminator output
        fake_output = self.discriminator(fake_data)

        # Generator loss with entropy regularization
        g_loss, loss_dict = self.loss_fn.generator_loss(fake_output, state_vectors)

        # Backward
        g_loss.backward()

        if self.gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.gradient_clip)

        self.g_optimizer.step()

        # Combine metrics
        metrics.update(loss_dict)

        return metrics

    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int = 1000,
        eval_every: int = 10,
        save_every: int = 100,
        log_every: int = 10,
        mode_centers: Optional[np.ndarray] = None,
    ):
        """
        Full training loop.

        Args:
            dataloader: Training data loader
            n_epochs: Number of epochs
            eval_every: Evaluate every N epochs
            save_every: Save checkpoint every N epochs
            log_every: Log metrics every N steps
            mode_centers: Mode centers for mode coverage evaluation
        """
        noise_dim = self.config.get("data", {}).get("noise_dim", 4)

        print(f"Starting training for {n_epochs} epochs...")
        print(f"Generator params: {self.generator.get_n_parameters()}")
        print(f"Discriminator params: {self.discriminator.get_n_parameters()}")

        global_step = 0

        for epoch in range(n_epochs):
            epoch_metrics = {
                "g_loss": [],
                "d_loss": [],
                "entropy_value": [],
            }

            # Training loop
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for batch_idx, real_data in enumerate(pbar):
                # Training step
                metrics = self.train_step(real_data, noise_dim=noise_dim)

                # Collect metrics
                epoch_metrics["g_loss"].append(metrics["total"])
                epoch_metrics["d_loss"].append(metrics["d_loss"])
                epoch_metrics["entropy_value"].append(metrics["entropy_value"])

                # Update progress bar
                pbar.set_postfix({
                    "G": f"{metrics['total']:.3f}",
                    "D": f"{metrics['d_loss']:.3f}",
                    "S": f"{metrics['entropy_value']:.3f}",
                })

                # Logging
                if global_step % log_every == 0 and self.use_wandb:
                    self.wandb.log(metrics, step=global_step)

                global_step += 1

            # Epoch summary
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

            # Evaluation
            if (epoch + 1) % eval_every == 0 and mode_centers is not None:
                eval_metrics = self.evaluate(mode_centers, n_samples=5000, noise_dim=noise_dim)
                avg_metrics.update(eval_metrics)

                print(
                    f"Epoch {epoch+1}: "
                    f"G={avg_metrics['g_loss']:.3f}, "
                    f"D={avg_metrics['d_loss']:.3f}, "
                    f"S={avg_metrics['entropy_value']:.3f}, "
                    f"MC={eval_metrics.get('mode_coverage', 0):.3f}, "
                    f"Div={eval_metrics.get('diversity', 0):.3f}"
                )

                # Log to WandB
                if self.use_wandb:
                    self.wandb.log({"epoch": epoch + 1, **avg_metrics})

            # Save history
            self.history["g_loss"].append(avg_metrics["g_loss"])
            self.history["d_loss"].append(avg_metrics["d_loss"])
            self.history["entropy"].append(avg_metrics["entropy_value"])

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch + 1)

        print("Training complete!")

    @torch.no_grad()
    def evaluate(
        self,
        mode_centers: np.ndarray,
        n_samples: int = 5000,
        noise_dim: int = 4,
    ) -> Dict[str, float]:
        """
        Evaluate the generator.

        Args:
            mode_centers: Mode centers for mode coverage
            n_samples: Number of samples to generate
            noise_dim: Noise dimension

        Returns:
            Dictionary of evaluation metrics
        """
        self.generator.eval()

        # Generate samples
        noise = torch.randn(n_samples, noise_dim, device=self.device)
        fake_data = self.generator(noise)
        fake_data_np = fake_data.cpu().numpy()

        # Compute metrics
        mode_coverage = compute_mode_coverage(
            fake_data_np,
            mode_centers,
            threshold=self.config.get("evaluation", {}).get("mode_coverage", {}).get("threshold", 0.3),
        )

        diversity = compute_diversity_score(fake_data_np)

        self.generator.train()

        return {
            "mode_coverage": mode_coverage,
            "diversity": diversity,
        }

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = f"{self.save_dir}/checkpoints/epoch_{epoch}.pt"

        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "g_optimizer_state_dict": self.g_optimizer.state_dict(),
                "d_optimizer_state_dict": self.d_optimizer.state_dict(),
                "history": self.history,
            },
            checkpoint_path,
        )

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        self.d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        self.history = checkpoint["history"]

        print(f"Checkpoint loaded: {checkpoint_path}")


__all__ = ["QGANTrainer"]
