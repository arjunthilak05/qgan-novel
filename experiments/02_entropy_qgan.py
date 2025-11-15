"""
Experiment: Entropy-Regularized QGAN

THE MAIN EXPERIMENT SCRIPT FOR YOUR NOVEL METHOD

Trains QGAN with entanglement entropy regularization on 8-Gaussian dataset.

Usage:
    python experiments/02_entropy_qgan.py

"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import numpy as np
from pathlib import Path

# Import our modules
from src.models import QuantumGenerator, Discriminator
from src.losses import EntropyRegularizedLoss
from src.training import QGANTrainer
from src.data import create_dataloader
from src.metrics import compute_mode_coverage, compute_diversity_score


def main():
    """Main experiment function."""

    print("=" * 60)
    print("ENTROPY-REGULARIZED QGAN EXPERIMENT")
    print("=" * 60)

    # ==========================================
    # 1. Load Configuration
    # ==========================================
    config_path = "configs/entropy_qgan.yaml"
    print(f"\nLoading config from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Alpha: {config['loss']['alpha']}")

    # Device
    device = config["experiment"]["device"]
    if not torch.cuda.is_available() and "cuda" in device:
        print("WARNING: CUDA not available, using CPU")
        device = "cpu"

    print(f"Device: {device}")

    # Set seed
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ==========================================
    # 2. Create Dataset
    # ==========================================
    print("\n" + "=" * 60)
    print("Creating dataset...")

    dataloader = create_dataloader(
        dataset_name=config["data"]["name"],
        batch_size=config["training"]["batch_size"],
        n_samples=config["data"]["train_size"],
        std=config["data"]["gaussian"]["std"],
        radius=config["data"]["gaussian"]["radius"],
    )

    # Get mode centers for evaluation
    dataset = dataloader.dataset
    mode_centers = dataset.get_mode_centers()

    print(f"Dataset: {config['data']['name']}")
    print(f"Train samples: {len(dataset)}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Number of modes: {len(mode_centers)}")

    # ==========================================
    # 3. Create Models
    # ==========================================
    print("\n" + "=" * 60)
    print("Creating models...")

    # Generator
    generator = QuantumGenerator(
        n_qubits=config["model"]["generator"]["n_qubits"],
        circuit_depth=config["model"]["generator"]["circuit_depth"],
        output_dim=2,  # 2D data
        ansatz=config["model"]["generator"]["ansatz"],
        simulator=config["model"]["generator"]["simulator"],
        device=device,
    )

    print(f"Generator: {generator.get_n_parameters()} parameters")
    print(f"Qubits: {config['model']['generator']['n_qubits']}")
    print(f"Depth: {config['model']['generator']['circuit_depth']}")
    print(f"Ansatz: {config['model']['generator']['ansatz']}")

    # Discriminator
    discriminator = Discriminator(
        input_dim=2,
        hidden_dims=config["model"]["discriminator"]["hidden_dims"],
        activation=config["model"]["discriminator"]["activation"],
        leaky_slope=config["model"]["discriminator"]["leaky_slope"],
        dropout=config["model"]["discriminator"]["dropout"],
        output_activation=config["model"]["discriminator"]["output_activation"],
    )

    print(f"Discriminator: {discriminator.get_n_parameters()} parameters")

    # ==========================================
    # 4. Create Loss Function
    # ==========================================
    print("\n" + "=" * 60)
    print("Creating loss function...")

    loss_fn = EntropyRegularizedLoss(
        alpha=config["loss"]["alpha"],
        partition_size=config["loss"]["entropy"]["partition_size"],
        entropy_type=config["loss"]["entropy"]["type"],
        base="2",
    )

    print(f"Loss type: {config['loss']['type']}")
    print(f"Alpha (entropy regularization): {config['loss']['alpha']}")
    print(f"Entropy type: {config['loss']['entropy']['type']}")
    print(f"Partition size: {config['loss']['entropy']['partition_size']}")

    # ==========================================
    # 5. Initialize WandB (optional)
    # ==========================================
    if config["logging"]["use_wandb"]:
        try:
            import wandb

            wandb.init(
                project=config["logging"]["wandb_project"],
                entity=config["logging"]["wandb_entity"],
                name=config["experiment"]["name"],
                config=config,
            )
            print("\nWandB logging enabled")
        except ImportError:
            print("\nWandB not installed, skipping")
            config["logging"]["use_wandb"] = False

    # ==========================================
    # 6. Create Trainer
    # ==========================================
    print("\n" + "=" * 60)
    print("Creating trainer...")

    trainer = QGANTrainer(
        generator=generator,
        discriminator=discriminator,
        loss_fn=loss_fn,
        device=device,
        config=config,
    )

    # ==========================================
    # 7. Train
    # ==========================================
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train(
        dataloader=dataloader,
        n_epochs=config["training"]["n_epochs"],
        eval_every=config["training"]["eval_every"],
        save_every=config["training"]["save_every"],
        log_every=config["training"]["log_every"],
        mode_centers=mode_centers,
    )

    # ==========================================
    # 8. Final Evaluation
    # ==========================================
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    final_metrics = trainer.evaluate(
        mode_centers=mode_centers,
        n_samples=config["evaluation"]["n_samples"],
        noise_dim=config["data"]["noise_dim"],
    )

    print(f"\nFinal Results:")
    print(f"  Mode Coverage: {final_metrics['mode_coverage']:.2%}")
    print(f"  Diversity Score: {final_metrics['diversity']:.3f}")

    # ==========================================
    # 9. Save Final Model
    # ==========================================
    final_checkpoint = f"{config['logging']['save_dir']}/checkpoints/final_model.pt"
    trainer.save_checkpoint(epoch=config["training"]["n_epochs"])

    print(f"\nFinal model saved to: {final_checkpoint}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)

    return trainer, final_metrics


if __name__ == "__main__":
    trainer, metrics = main()
