# tests/test_ml/test_features/test_extractors.py
import pytest
import numpy as np
import librosa
from src.ml.features.extractors import (
    BASIC_FEATURES,
    SPECTRAL_FEATURES,
    RHYTHM_FEATURES
)


@pytest.fixture
def test_audio() -> np.ndarray:
    """Generate a test audio signal for testing features

    Returns:
        np.ndarray: Synthetic audio signal (1 second of 440Hz sine wave)
    """
    duration = 1.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def sample_rate() -> int:
    """Sample rate for test audio"""
    return 22050


class TestBasicFeatures:
    """Test suite for basic audio features"""

    def test_zcr(self, test_audio):
        """Test zero crossing rate extraction"""
        zcr = BASIC_FEATURES['zcr'](test_audio)
        assert isinstance(zcr, (float, np.float32, np.float64))
        assert 0 <= zcr <= 1  # ZCR should be normalized between 0 and 1

    def test_energy(self, test_audio):
        """Test energy extraction"""
        energy = BASIC_FEATURES['energy'](test_audio)
        assert isinstance(energy, (float, np.float32, np.float64))
        assert energy >= 0  # Energy should be non-negative

    def test_rms(self, test_audio):
        """Test RMS extraction"""
        rms = BASIC_FEATURES['rms'](test_audio)
        assert isinstance(rms, (float, np.float32, np.float64))
        assert rms >= 0  # RMS should be non-negative


class TestSpectralFeatures:
    """Test suite for spectral features"""

    def test_spectral_centroid(self, test_audio, sample_rate):
        """Test spectral centroid extraction"""
        centroid = SPECTRAL_FEATURES['spectral_centroid'](test_audio, sample_rate)
        assert isinstance(centroid, (float, np.float32, np.float64))
        assert 0 <= centroid <= sample_rate / 2  # Should be within Nyquist frequency