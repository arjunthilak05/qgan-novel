"""
Loss Functions for ER-QGAN
"""

from .gan_losses import (
    vanilla_gan_loss,
    wasserstein_loss,
    hinge_loss,
)
from .entropy_regularized import (
    EntropyRegularizedLoss,
    AdaptiveAlphaScheduler,
)

__all__ = [
    "vanilla_gan_loss",
    "wasserstein_loss",
    "hinge_loss",
    "EntropyRegularizedLoss",
    "AdaptiveAlphaScheduler",
]
