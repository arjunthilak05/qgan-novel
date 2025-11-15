"""
Standard GAN Loss Functions

Baseline loss functions without entropy regularization.
"""

import torch
import torch.nn as nn
from typing import Tuple


def vanilla_gan_loss(
    real_output: torch.Tensor,
    fake_output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard vanilla GAN loss (binary cross-entropy).

    Discriminator: max log(D(x)) + log(1 - D(G(z)))
    Generator: max log(D(G(z))) = min -log(D(G(z)))

    Args:
        real_output: Discriminator output for real data (batch_size, 1)
        fake_output: Discriminator output for fake data (batch_size, 1)

    Returns:
        (d_loss, g_loss)
    """
    # Discriminator loss
    real_loss = -torch.log(real_output + 1e-8).mean()
    fake_loss = -torch.log(1 - fake_output + 1e-8).mean()
    d_loss = real_loss + fake_loss

    # Generator loss (non-saturating version)
    g_loss = -torch.log(fake_output + 1e-8).mean()

    return d_loss, g_loss


def wasserstein_loss(
    real_output: torch.Tensor,
    fake_output: torch.Tensor,
    gradient_penalty: torch.Tensor = None,
    lambda_gp: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wasserstein GAN loss (WGAN-GP).

    Discriminator: max E[D(x)] - E[D(G(z))] - λ·GP
    Generator: max E[D(G(z))]

    Args:
        real_output: Critic output for real data
        fake_output: Critic output for fake data
        gradient_penalty: Gradient penalty term
        lambda_gp: Weight for gradient penalty

    Returns:
        (d_loss, g_loss)
    """
    # Discriminator/Critic loss
    d_loss = -real_output.mean() + fake_output.mean()

    if gradient_penalty is not None:
        d_loss += lambda_gp * gradient_penalty

    # Generator loss
    g_loss = -fake_output.mean()

    return d_loss, g_loss


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP.

    GP = E[(||∇_x D(x)||_2 - 1)^2]

    where x is sampled uniformly along lines between real and fake data.

    Args:
        discriminator: Discriminator/Critic network
        real_data: Real data samples
        fake_data: Fake data samples
        device: Device

    Returns:
        Gradient penalty scalar
    """
    batch_size = real_data.size(0)

    # Random interpolation weight
    alpha = torch.rand(batch_size, 1, device=device)

    # Interpolated samples
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    # Discriminator output
    d_interpolates = discriminator(interpolates)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten gradients
    gradients = gradients.view(batch_size, -1)

    # Compute gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


def hinge_loss(
    real_output: torch.Tensor,
    fake_output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hinge loss for GANs.

    Discriminator: max min(0, -1 + D(x)) + min(0, -1 - D(G(z)))
    Generator: max D(G(z))

    Args:
        real_output: Discriminator output for real data
        fake_output: Discriminator output for fake data

    Returns:
        (d_loss, g_loss)
    """
    # Discriminator loss
    real_loss = torch.nn.functional.relu(1.0 - real_output).mean()
    fake_loss = torch.nn.functional.relu(1.0 + fake_output).mean()
    d_loss = real_loss + fake_loss

    # Generator loss
    g_loss = -fake_output.mean()

    return d_loss, g_loss


__all__ = [
    "vanilla_gan_loss",
    "wasserstein_loss",
    "compute_gradient_penalty",
    "hinge_loss",
]
