"""
Entropy-Regularized Quantum GAN (ER-QGAN)

A novel approach to improving mode coverage in Quantum GANs
by explicitly optimizing entanglement entropy during training.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from . import models
from . import entropy
from . import losses
from . import training
from . import data
from . import metrics
from . import utils

__all__ = [
    "models",
    "entropy",
    "losses",
    "training",
    "data",
    "metrics",
    "utils",
]
