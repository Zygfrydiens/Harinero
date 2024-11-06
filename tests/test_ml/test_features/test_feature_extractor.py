import pytest
import numpy as np
from src.ml.features.extractors import (
    BASIC_FEATURES,
    SPECTRAL_FEATURES,
    RHYTHM_FEATURES,
    FeatureExtractor
)


@pytest.fixture
def feature_extractor(test_config):
    """Create FeatureExtractor instance using project test configuration"""
    return FeatureExtractor(test_config)


@pytest.fixture
def test_audio(test_config):
    """Generate test audio signal suitable for rhythm analysis

    Creates a signal with clear rhythmic pattern by combining multiple frequencies
    and adding amplitude modulation for beat-like effects.
    """
    sr = test_config['data']['sampling_rate']
    duration = 3.0  # Longer duration for better rhythm analysis
    t = np.linspace(0, duration, int(sr * duration))

    # Create base signal with multiple frequencies
    signal = np.sin(2 * np.pi * 100 * t)  # Base frequency
    signal += 0.5 * np.sin(2 * np.pi * 200 * t)  # Harmonic

    # Add amplitude modulation to create rhythm (2 Hz modulation = 120 BPM)
    beat_freq = 2
    envelope = 0.5 * (1 + np.sin(2 * np.pi * beat_freq * t))
    signal = signal * envelope

    # Normalize
    signal = signal / np.max(np.abs(signal))

    return signal


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class"""

    def test_initialization(self, feature_extractor, test_config):
        """Test proper initialization of FeatureExtractor"""
        assert feature_extractor.sampling_rate == test_config['data']['sampling_rate']
        assert hasattr(feature_extractor, 'basic_extractors')
        assert hasattr(feature_extractor, 'spectral_extractors')
        assert hasattr(feature_extractor, 'rhythm_extractors')

    def test_extract_basic_features(self, feature_extractor, test_audio):
        """Test extraction of basic feature group"""
        features = feature_extractor.extract_feature_group(test_audio, group='basic')
        assert isinstance(features, dict)
        assert all(key in features for key in ['zcr', 'energy', 'rms'])
        assert all(isinstance(v, (float, np.float32, np.float64)) for v in features.values())

    def test_extract_spectral_features(self, feature_extractor, test_audio):
        """Test extraction of spectral feature group"""
        features = feature_extractor.extract_feature_group(test_audio, group='spectral')
        assert isinstance(features, dict)
        assert 'spectral_centroid' in features
        assert all(isinstance(v, (float, np.float32, np.float64)) for v in features.values())

    def test_invalid_group(self, feature_extractor, test_audio):
        """Test handling of invalid feature group"""
        with pytest.raises(ValueError, match="Unknown group"):
            feature_extractor.extract_feature_group(test_audio, group='invalid_group')

    @pytest.mark.parametrize("audio_length", [0, 100, 1000])
    def test_different_audio_lengths(self, feature_extractor, audio_length):
        """Test feature extraction with different audio lengths"""
        audio = np.zeros(audio_length)
        if audio_length == 0:
            with pytest.raises(ValueError):
                feature_extractor.extract_feature_group(audio, group='basic')
        else:
            features = feature_extractor.extract_feature_group(audio, group='basic')
            assert isinstance(features, dict)

    def test_all_feature_groups(self, feature_extractor, test_audio):
        """Test extraction of all feature groups"""
        features = feature_extractor.extract_feature_group(test_audio, group='all')
        assert isinstance(features, dict)
        # Check if we have features from all groups
        assert any(key in features for key in BASIC_FEATURES)
        assert any(key in features for key in SPECTRAL_FEATURES)
        assert any(key in features for key in RHYTHM_FEATURES)
