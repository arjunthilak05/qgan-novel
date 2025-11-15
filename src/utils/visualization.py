"""
Visualization Utilities for Paper Figures

Functions to create publication-quality plots for the paper.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
import torch

# Set publication-quality defaults
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "figure.figsize": (8, 6),
    "figure.dpi": 150,
})


def plot_2d_samples(
    real_samples: np.ndarray,
    fake_samples: np.ndarray,
    mode_centers: Optional[np.ndarray] = None,
    title: str = "Generated vs Real Samples",
    save_path: Optional[str] = None,
):
    """
    Plot 2D scatter plot of real vs fake samples.

    Args:
        real_samples: Real data points (n, 2)
        fake_samples: Generated data points (n, 2)
        mode_centers: Optional mode centers to highlight
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Real data
    axes[0].scatter(real_samples[:, 0], real_samples[:, 1], alpha=0.5, s=10, label="Real")
    if mode_centers is not None:
        axes[0].scatter(
            mode_centers[:, 0],
            mode_centers[:, 1],
            color="red",
            marker="x",
            s=200,
            linewidths=3,
            label="Mode Centers",
        )
    axes[0].set_title("Real Data Distribution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].set_aspect("equal")

    # Fake data
    axes[1].scatter(fake_samples[:, 0], fake_samples[:, 1], alpha=0.5, s=10, label="Generated", color="orange")
    if mode_centers is not None:
        axes[1].scatter(
            mode_centers[:, 0],
            mode_centers[:, 1],
            color="red",
            marker="x",
            s=200,
            linewidths=3,
            label="Mode Centers",
        )
    axes[1].set_title("Generated Data Distribution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].legend()
    axes[1].set_aspect("equal")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_mode_coverage_comparison(
    methods: List[str],
    mode_coverages: List[float],
    errors: Optional[List[float]] = None,
    title: str = "Mode Coverage Comparison",
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing mode coverage across methods.

    Args:
        methods: List of method names
        mode_coverages: List of mode coverage values
        errors: Optional error bars (std or confidence intervals)
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(methods, mode_coverages, color=colors[: len(methods)], alpha=0.8)

    if errors is not None:
        ax.errorbar(
            methods,
            mode_coverages,
            yerr=errors,
            fmt="none",
            ecolor="black",
            capsize=5,
        )

    ax.set_ylabel("Mode Coverage (%)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([0, 1.0])
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Perfect Coverage")

    # Add value labels on bars
    for bar, coverage in zip(bars, mode_coverages):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{coverage:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
):
    """
    Plot training curves (losses, entropy, mode coverage).

    Args:
        history: Dictionary with keys: 'g_loss', 'd_loss', 'entropy', 'mode_coverage'
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generator Loss
    axes[0, 0].plot(history["g_loss"], label="Generator Loss", color="blue")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Generator Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Discriminator Loss
    axes[0, 1].plot(history["d_loss"], label="Discriminator Loss", color="orange")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Discriminator Loss")
    axes[0, 1].grid(True, alpha=0.3)

    # Entanglement Entropy
    if "entropy" in history and len(history["entropy"]) > 0:
        axes[1, 0].plot(history["entropy"], label="Entanglement Entropy", color="green")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("S(ρ_A)")
        axes[1, 0].set_title("Entanglement Entropy")
        axes[1, 0].grid(True, alpha=0.3)

    # Mode Coverage
    if "mode_coverage" in history and len(history["mode_coverage"]) > 0:
        axes[1, 1].plot(history["mode_coverage"], label="Mode Coverage", color="red")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Coverage (%)")
        axes[1, 1].set_title("Mode Coverage Over Training")
        axes[1, 1].set_ylim([0, 1.0])
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_entropy_vs_mode_coverage(
    entropies: List[float],
    mode_coverages: List[float],
    alphas: Optional[List[float]] = None,
    title: str = "Entropy vs Mode Coverage",
    save_path: Optional[str] = None,
):
    """
    Scatter plot showing correlation between entropy and mode coverage.

    Args:
        entropies: List of entropy values
        mode_coverages: List of mode coverage values
        alphas: Optional list of alpha values for color coding
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if alphas is not None:
        scatter = ax.scatter(
            entropies,
            mode_coverages,
            c=alphas,
            cmap="viridis",
            s=100,
            alpha=0.7,
            edgecolors="black",
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Alpha (α)", fontsize=12)
    else:
        ax.scatter(entropies, mode_coverages, s=100, alpha=0.7, edgecolors="black")

    # Fit linear regression
    if len(entropies) > 1:
        z = np.polyfit(entropies, mode_coverages, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(entropies), max(entropies), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, label=f"Fit: y={z[0]:.2f}x+{z[1]:.2f}")

    ax.set_xlabel("Entanglement Entropy S(ρ_A)", fontsize=13)
    ax.set_ylabel("Mode Coverage (%)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_ablation_heatmap(
    alphas: List[float],
    partitions: List[int],
    mode_coverages: np.ndarray,
    title: str = "Ablation Study: Alpha vs Partition Size",
    save_path: Optional[str] = None,
):
    """
    Heatmap for ablation study results.

    Args:
        alphas: List of alpha values
        partitions: List of partition sizes
        mode_coverages: 2D array of mode coverage values (alphas x partitions)
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        mode_coverages,
        annot=True,
        fmt=".1%",
        cmap="YlGnBu",
        xticklabels=partitions,
        yticklabels=[f"{a:.2f}" for a in alphas],
        cbar_kws={"label": "Mode Coverage (%)"},
        ax=ax,
    )

    ax.set_xlabel("Partition Size", fontsize=13)
    ax.set_ylabel("Alpha (α)", fontsize=13)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


__all__ = [
    "plot_2d_samples",
    "plot_mode_coverage_comparison",
    "plot_training_curves",
    "plot_entropy_vs_mode_coverage",
    "plot_ablation_heatmap",
]
