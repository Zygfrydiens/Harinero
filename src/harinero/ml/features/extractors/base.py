from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import numpy as np

ConfigType = TypeVar('ConfigType')


class BaseFeatureExtractor(ABC, Generic[ConfigType]):
    """Abstract base class for all feature extractors.

    Reason:
        Provides a consistent interface for all feature extraction implementations
        while enforcing proper configuration and extraction patterns.
    """

    def __init__(self, config: ConfigType):
        """Initialize feature extractor with configuration.

        Args:
            config: Configuration object/dict specific to the extractor type

        Raises:
            ValueError: If config validation fails
        """
        self.validate_config(config)
        self.config = config

    @abstractmethod
    def validate_config(self, config: ConfigType) -> None:
        """Validate extractor configuration.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @abstractmethod
    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract features from audio data.

        Args:
            audio: Audio signal array
            sample_rate: Sampling rate of the audio

        Returns:
            Extracted features as numpy array

        Raises:
            ValueError: If input data is invalid
        """
        pass