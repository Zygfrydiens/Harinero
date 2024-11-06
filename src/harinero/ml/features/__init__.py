"""
Feature engineering and processing components.

This package provides functionality for extracting and processing audio features
for the Harinero tanda creator project.
"""

from .feature_processor import (
    process_song_metadata,
    extract_and_add_features,
    initialize_feature_columns,
    save_df,
    FEATURE_CONFIG
)

__all__ = [
    'process_song_metadata',
    'extract_and_add_features',
    'initialize_feature_columns',
    'save_df',
    'FEATURE_CONFIG'
]