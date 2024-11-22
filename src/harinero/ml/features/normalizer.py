from typing import Dict, Optional
import numpy as np


class FeatureNormalizer:
    """Handles feature normalization for audio processing pipeline.

    Reason:
        Provides consistent feature normalization across different feature types
        while maintaining statistics for reproducibility.
    """

    def __init__(self):
        self._stats: Optional[Dict[str, np.ndarray]] = None

    def fit(self, features: np.ndarray) -> None:
        """Calculate normalization statistics from feature array.

        Args:
            features: Array of shape (n_samples, n_features) to fit normalizer on

        Note:
            Handles zero standard deviation by setting it to 1.0 to avoid division by zero
        """
        self._stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0)
        }
        # Handle zero std
        self._stats['std'][self._stats['std'] == 0] = 1.0

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using stored statistics.

        Args:
            features: Feature array to normalize

        Returns:
            Normalized features array

        Raises:
            ValueError: If normalizer hasn't been fitted
        """
        if self._stats is None:
            raise ValueError("Normalizer not fitted! Call fit() first.")

        return (features - self._stats['mean']) / self._stats['std']

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Convenience method to fit and transform in one call.

        Args:
            features: Feature array to fit and normalize

        Returns:
            Normalized features array
        """
        self.fit(features)
        return self.transform(features)

    def get_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Get current normalization statistics.

        Returns:
            Dictionary with 'mean' and 'std' arrays or None if not fitted
        """
        return self._stats.copy() if self._stats is not None else None