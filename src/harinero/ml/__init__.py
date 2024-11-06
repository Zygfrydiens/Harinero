"""Machine learning components for the Harinero Tanda project.

This package provides the core ML functionality including:
- Audio preprocessing and feature extraction
- Model implementations (Keras, PyTorch)
- Clustering analysis
- Feature engineering

The components work together to provide a complete pipeline for
learning and analyzing tango music similarities.
"""

from typing import Dict, List, Optional, Union, Any

# Import main components from submodules
from .preprocessing import Preprocessor
from .models import KerasSongModel, BaseModel
from .clustering import (
    perform_kmeans_clustering,
    prepare_plot_parameters,
    plot_pca,
    evaluate_clustering_performance,
)
from .features import (
    process_song_metadata,
    extract_and_add_features,
    initialize_feature_columns,
    FEATURE_CONFIG
)

# Define package-level exports
__all__ = [
    # Preprocessing
    'Preprocessor',

    # Models
    'KerasSongModel',
    'BaseModel',

    # Clustering
    'perform_kmeans_clustering',
    'prepare_plot_parameters',
    'plot_pca',
    'evaluate_clustering_performance',

    # Feature Engineering
    'process_song_metadata',
    'extract_and_add_features',
    'initialize_feature_columns',
    'FEATURE_CONFIG',
]

# Type aliases that might be useful across the ML package
ModelOutput = Dict[str, Union[float, List[float]]]
FeatureVector = Dict[str, float]
AudioFeatures = Dict[str, Any]
