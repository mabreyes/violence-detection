"""Violence Detection System for Philippine Context with Limited Compute Resources.

This package provides tools for detecting violence in videos using a hybrid
4D CNN + LSTM + Transformer architecture, optimized for limited computational resources
and fine-tuned for the Philippine context.
"""

import logging

from .__version__ import __version__

# Set up package-level logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "__version__",
]
