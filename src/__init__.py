"""Harinero Tanda Creator

A machine learning system for analyzing similarity between Argentine tango songs
and creating tandas based on musical similarity metrics.

Components:
    - Core: Basic data structures and database operations
    - ML: Machine learning models and feature engineering
    - Tools: Utility functions and auxiliary tools
"""

__version__ = "0.1.0"

# Core structures
from .core.models.structures import SongStruct, TandaStruct, MilongaStruct

# ML components
from .ml.features.extractors import (
    BASIC_FEATURES,
    SPECTRAL_FEATURES,
    RHYTHM_FEATURES,
    AVAILABLE_EXTRACTORS,
    FeatureExtractor
)
from .ml.features.encoders import (
    encode_feature_names,
    decode_feature_names,
    feature_groups
)

# Make commonly used components available at package level
__all__ = [
    # Core structures
    'SongStruct',
    'TandaStruct',
    'MilongaStruct',

    # Feature extraction
    'BASIC_FEATURES',
    'SPECTRAL_FEATURES',
    'RHYTHM_FEATURES',
    'AVAILABLE_EXTRACTORS',
    'FeatureExtractor',

    # Feature encoding
    'encode_feature_names',
    'decode_feature_names',
    'feature_groups',
]