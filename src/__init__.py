"""
adaptive-muon: A PyTorch project with enforced code quality.

This package demonstrates best practices for PyTorch development with
comprehensive linting and style checking.
"""

__version__ = "0.1.0"

# Re-export commonly used classes from train_model for convenient imports
from .train_model import CifarLoader, CifarNet  # noqa: F401

__all__ = ["CifarLoader", "CifarNet"]
