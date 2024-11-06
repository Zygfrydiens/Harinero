from typing import List, Dict

"""
Feature Engineering Package for Harinero Tanda Creator

This package provides functionality for audio feature extraction, encoding, and processing
for the Harinero tanda similarity analysis system.

Examples:
    Basic feature encoding:
    >>> from feature_engineering import encode_feature_names
    >>> encode_feature_names(['spectral_centroid'])
    ['f1']

    Accessing feature groups:
    >>> from feature_engineering import feature_groups
    >>> feature_groups['time_domain']
    ['zcr', 'energy', 'rms', ...]
"""

from .feature_encoding import (
    encode_feature_names,
    decode_feature_names,
    categories,
    feature_groups,
    original_to_encoded,
    encoded_to_original,
)

__all__ = [
    'encode_feature_names',
    'decode_feature_names',
    'categories',
    'feature_groups',
    'original_to_encoded',
    'encoded_to_original',
]