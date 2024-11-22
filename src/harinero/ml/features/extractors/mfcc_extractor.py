from dataclasses import dataclass
from typing import Optional
from typing import Dict, Any
import numpy as np
import librosa

from .base import BaseFeatureExtractor, ConfigType


class MFCCExtractor(BaseFeatureExtractor[Dict[str, Any]]):
    """MFCC feature extractor with optional delta features.

    Reason:
        Extracts MFCC features from audio data with configurable parameters
        and optional delta/delta-delta calculations for capturing temporal dynamics.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize MFCC extractor with configuration.

        Args:
            config: Configuration dictionary containing features.mfcc section with:
                - n_mfcc: Number of MFCC coefficients
                - n_fft: FFT window size in samples
                - hop_length: Number of samples between frames
                - n_mels: Number of mel bands
                - fmin: Minimum frequency
                - fmax: Maximum frequency (optional)
                - include_deltas: Whether to include delta features

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)

        mfcc_config = config['features']['mfcc']
        self._n_mfcc = mfcc_config['n_mfcc']
        self._n_fft = mfcc_config['n_fft']
        self._hop_length = mfcc_config['hop_length']
        self._n_mels = mfcc_config['n_mels']
        self._fmin = mfcc_config['fmin']
        self._fmax = mfcc_config.get('fmax')  # Optional
        self._include_deltas = mfcc_config.get('include_deltas', True)  # Default to True

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate MFCC configuration parameters.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        if 'features' not in config or 'mfcc' not in config['features']:
            raise ValueError("Config must contain 'features.mfcc' section")

        mfcc_config = config['features']['mfcc']
        required_params = ['n_mfcc', 'n_fft', 'hop_length', 'n_mels', 'fmin']

        for param in required_params:
            if param not in mfcc_config:
                raise ValueError(f"Missing required parameter: {param}")
            if mfcc_config[param] <= 0:
                raise ValueError(f"Parameter {param} must be positive")

        if 'fmax' in mfcc_config and mfcc_config['fmax'] is not None:
            if mfcc_config['fmax'] <= mfcc_config['fmin']:
                raise ValueError("fmax must be greater than fmin")

    def extract(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Extract MFCC features from audio.

        Args:
            audio: Audio signal array
            sample_rate: Sampling rate of the audio

        Returns:
            Array of shape (n_frames, n_features) where n_features is:
            - n_mfcc if include_deltas is False
            - n_mfcc * 3 if include_deltas is True (mfcc + delta + delta2)

        Raises:
            ValueError: If audio data is invalid
        """
        if len(audio) == 0:
            raise ValueError("Audio array is empty")

        # Calculate fmax if not provided
        fmax = self._fmax or sample_rate // 2

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=self._n_mfcc,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            n_mels=self._n_mels,
            fmin=self._fmin,
            fmax=fmax
        )

        if not self._include_deltas:
            return mfcc.T

        # Calculate deltas if requested
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # Stack and reshape features
        features = np.vstack([mfcc, delta, delta2])
        return features.T