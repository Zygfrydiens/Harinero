import pytest
import numpy as np
import librosa


@pytest.fixture(scope="session")
def test_audio_file(tmp_path_factory):
    """Create a temporary audio file for testing"""
    duration = 1.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * 440 * t)

    # Create temporary file
    temp_dir = tmp_path_factory.mktemp("audio")
    file_path = temp_dir / "test_audio.wav"
    librosa.output.write_wav(str(file_path), y, sr)

    return file_path