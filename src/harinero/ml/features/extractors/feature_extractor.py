from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
from .features import (
    AVAILABLE_EXTRACTORS,
    BASIC_FEATURES,
    SPECTRAL_FEATURES,
    RHYTHM_FEATURES,
    extract_statistical_perceptual_features,
    extract_advanced_features
)

from typing import Dict, Any, Callable, Union
import numpy as np


class FeatureExtractor:
    """Audio feature extractor with configurable parameters"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sampling_rate = config['data']['sampling_rate']

        # Feature groups remain the same
        self.basic_extractors = BASIC_FEATURES
        self.spectral_extractors = SPECTRAL_FEATURES
        self.rhythm_extractors = RHYTHM_FEATURES
        self.all_extractors = AVAILABLE_EXTRACTORS

    def _execute_feature(self,
                         func: Callable,
                         audio: np.ndarray,
                         name: str) -> Union[float, np.ndarray]:
        """Execute feature extraction function with appropriate parameters

        Args:
            func: Feature extraction function
            audio: Audio signal
            name: Feature name (used for special cases)

        Returns:
            Extracted feature value
        """
        # Get function parameters
        from inspect import signature
        params = signature(func).parameters

        # If function accepts sr parameter, pass it
        if 'sr' in params:
            return func(audio, sr=self.sampling_rate)
        return func(audio)

    def extract_feature_group(self,
                              audio: np.ndarray,
                              group: str = 'basic') -> Dict[str, float]:
        """Extract all features from a specific group

        Args:
            audio: Audio signal as numpy array
            group: Feature group to extract ('basic', 'spectral', 'rhythm', or 'all')

        Returns:
            Dictionary of features where keys are feature names and values are floats

        Raises:
            ValueError: If group is unknown
        """
        groups = {
            'basic': self.basic_extractors,
            'spectral': self.spectral_extractors,
            'rhythm': self.rhythm_extractors,
            'all': self.all_extractors
        }

        if group not in groups:
            raise ValueError(f"Unknown group: {group}")

        extractors = groups[group]
        return {
            name: self._execute_feature(func, audio, name)
            for name, func in extractors.items()
        }