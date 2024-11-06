"""
Clustering module for analyzing tango song similarities.

This module provides functionality for clustering analysis of tango songs
based on their extracted features.
"""

from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd

from .clustering import (
    perform_kmeans_clustering,
    prepare_plot_parameters,
    plot_pca,
    evaluate_clustering_performance,
)

__all__ = [
    'perform_kmeans_clustering',
    'prepare_plot_parameters',
    'plot_pca',
    'evaluate_clustering_performance',
]

# Type aliases for better code readability
ClusteringResult = Tuple[pd.DataFrame, np.ndarray, np.ndarray]
PlotParams = Dict[str, Union[List[str], List[float], int, str]]