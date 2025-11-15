"""
Experiment: Baseline QGAN (No Entropy Regularization)

Standard QGAN without entropy regularization for comparison.

Usage:
    python experiments/01_baseline_qgan.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import numpy as np

from src.models import QuantumGenerator, Discriminator
from src.losses import EntropyRegularizedLoss  # alpha=0 for baseline
from src.training import QGANTrainer
from src.data import create_dataloader


def main():
    print("=" * 60)
    print("BASELINE QGAN EXPERIMENT (NO ENTROPY)")
    print("=" * 60)

    # Load config
    config_path = "configs/baseline_qgan.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"\nExperiment: {config['experiment']['name']}")
    print(f"Alpha: {config['loss']['alpha']} (NO REGULARIZATION)")

    # Device
    device = config["experiment"]["device"]
    if not torch.cuda.is_available() and "cuda" in device:
        device = "cpu"

    # Seed
    torch.manual_seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])

    # Dataset
    dataloader = create_dataloader(
        dataset_name=config["data"]["name"],
        batch_size=config["training"]["batch_size"],
        n_samples=config["data"]["train_size"],
        std=config["data"]["gaussian"]["std"],
        radius=config["data"]["gaussian"]["radius"],
    )

    mode_centers = dataloader.dataset.get_mode_centers()

    # Models
    generator = QuantumGenerator(
        n_qubits=config["model"]["generator"]["n_qubits"],
        circuit_depth=config["model"]["generator"]["circuit_depth"],
        output_dim=2,
        ansatz=config["model"]["generator"]["ansatz"],
        simulator=config["model"]["generator"]["simulator"],
        device=device,
    )

    discriminator = Discriminator(
        input_dim=2,
        hidden_dims=config["model"]["discriminator"]["hidden_dims"],
        activation=config["model"]["discriminator"]["activation"],
        leaky_slope=config["model"]["discriminator"]["leaky_slope"],
        dropout=config["model"]["discriminator"]["dropout"],
        output_activation=config["model"]["discriminator"]["output_activation"],
    )

    # Loss (alpha=0 for baseline)
    loss_fn = EntropyRegularizedLoss(alpha=0.0)  # No entropy regularization!

    # WandB
    if config["logging"]["use_wandb"]:
        try:
            import wandb
            wandb.init(
                project=config["logging"]["wandb_project"],
                name=config["experiment"]["name"],
                config=config,
            )
        except ImportError:
            config["logging"]["use_wandb"] = False

    # Trainer
    trainer = QGANTrainer(
        generator=generator,
        discriminator=discriminator,
        loss_fn=loss_fn,
        device=device,
        config=config,
    )

    # Train
    print("\nStarting training...")
    trainer.train(
        dataloader=dataloader,
        n_epochs=config["training"]["n_epochs"],
        eval_every=config["training"]["eval_every"],
        save_every=config["training"]["save_every"],
        log_every=config["training"]["log_every"],
        mode_centers=mode_centers,
    )

    # Final evaluation
    final_metrics = trainer.evaluate(
        mode_centers=mode_centers,
        n_samples=config["evaluation"]["n_samples"],
        noise_dim=config["data"]["noise_dim"],
    )

    print(f"\n{'='*60}")
    print("BASELINE RESULTS:")
    print(f"  Mode Coverage: {final_metrics['mode_coverage']:.2%}")
    print(f"  Diversity Score: {final_metrics['diversity']:.3f}")
    print(f"{'='*60}")

    return trainer, final_metrics


if __name__ == "__main__":
    trainer, metrics = main()
