import pytest
import numpy as np


@pytest.fixture
def sample_audio():
    """Generate sample audio data"""
    sample_rate = 22050
    duration = 3
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def feature_config():
    """Feature extraction configuration"""
    return {
        "frame_size": 2048,
        "hop_length": 512,
        "sample_rate": 22050,
        "features": ["mfcc", "spectral_contrast", "chroma"]
    }
