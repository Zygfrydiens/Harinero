"""Machine Learning model implementations for the Harinero Tanda project.

This module contains different model implementations (Keras and PyTorch)
for song similarity learning.
"""

from .model_keras import KerasSongModel
from .base import BaseModel

__all__ = ['KerasSongModel', 'BaseModel']