"""
Classical Discriminator Network

Standard feedforward neural network that classifies real vs. fake data.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class Discriminator(nn.Module):
    """
    Classical discriminator network.

    Args:
        input_dim: Dimension of input data
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'leaky_relu', 'tanh')
        leaky_slope: Slope for LeakyReLU
        dropout: Dropout probability
        output_activation: Output activation ('sigmoid' for standard GAN)
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dims: List[int] = [64, 32, 16],
        activation: str = "leaky_relu",
        leaky_slope: float = 0.2,
        dropout: float = 0.2,
        output_activation: str = "sigmoid",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build network
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(leaky_slope))
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "none":
            pass  # No output activation (for Wasserstein GAN)
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output probabilities of shape (batch_size, 1)
        """
        return self.network(x)

    def get_n_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


class ConvDiscriminator(nn.Module):
    """
    Convolutional discriminator for image data (e.g., MNIST).

    Args:
        image_size: Size of square input images (e.g., 28 for MNIST)
        n_channels: Number of input channels (1 for grayscale, 3 for RGB)
        n_filters: Base number of convolutional filters
        output_activation: Output activation
    """

    def __init__(
        self,
        image_size: int = 28,
        n_channels: int = 1,
        n_filters: int = 64,
        output_activation: str = "sigmoid",
    ):
        super().__init__()

        self.image_size = image_size
        self.n_channels = n_channels

        # Convolutional layers
        self.conv_net = nn.Sequential(
            # Layer 1: (1, 28, 28) -> (64, 14, 14)
            nn.Conv2d(n_channels, n_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Layer 2: (64, 14, 14) -> (128, 7, 7)
            nn.Conv2d(n_filters, n_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),
            # Layer 3: (128, 7, 7) -> (256, 3, 3)
            nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2),
        )

        # Calculate flattened size
        conv_output_size = n_filters * 4 * (image_size // 8) * (image_size // 8)

        # Fully connected layers
        fc_layers = [
            nn.Flatten(),
            nn.Linear(conv_output_size, 1),
        ]

        if output_activation == "sigmoid":
            fc_layers.append(nn.Sigmoid())

        self.fc_net = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, n_channels, height, width)

        Returns:
            Output probabilities of shape (batch_size, 1)
        """
        features = self.conv_net(x)
        output = self.fc_net(features)
        return output

    def get_n_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())


def create_discriminator(config: dict, device: str = "cuda:0") -> nn.Module:
    """
    Factory function to create discriminator from config.

    Args:
        config: Configuration dictionary
        device: Torch device

    Returns:
        Discriminator instance
    """
    disc_config = config["model"]["discriminator"]
    data_config = config["data"]

    # Determine input dimension from dataset
    if "mnist" in data_config["name"].lower():
        # Use convolutional discriminator for images
        discriminator = ConvDiscriminator(
            image_size=28,
            n_channels=1,
            n_filters=64,
            output_activation=disc_config.get("output_activation", "sigmoid"),
        )
    else:
        # Use MLP for low-dimensional data
        output_dim = 2  # Default for 2D distributions

        discriminator = Discriminator(
            input_dim=output_dim,
            hidden_dims=disc_config["hidden_dims"],
            activation=disc_config["activation"],
            leaky_slope=disc_config.get("leaky_slope", 0.2),
            dropout=disc_config.get("dropout", 0.2),
            output_activation=disc_config.get("output_activation", "sigmoid"),
        )

    return discriminator.to(device)
