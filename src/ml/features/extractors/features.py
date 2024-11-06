from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import librosa

__all__ = [
    # Basic features
    'extract_zcr',
    'extract_energy',
    'extract_rms',
    'extract_autocorrelation',

    # Spectral features
    'extract_spectral_centroid',
    'extract_spectral_bandwidth',
    'extract_spectral_rolloff',
    'extract_spectral_contrast',
    'extract_spectral_flux',

    # Advanced features
    'extract_mfcc',
    'extract_chromagram',
    'extract_mel_spectrogram',
    'extract_tonnetz',
    'extract_temporal_dynamics',

    # Rhythm features
    'extract_rhythm_tempo',
    'extract_beat_strength',
    'extract_onset_rate',

    # High-level feature sets
    'extract_statistical_perceptual_features',
    'extract_advanced_features',
    'extract_regularity_features'
]

#TODO: brak this down into smaller file


def extract_zcr(y):
    """Extract Zero Crossing Rate"""
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.mean(zcr)


def extract_energy(y):
    """Extract Energy"""
    energy = np.sum(y ** 2) / len(y)
    return energy


def extract_rms(y):
    """Extract RMS Amplitude"""
    rms = librosa.feature.rms(y=y)
    return np.mean(rms)


def extract_autocorrelation(y):
    """Extract Autocorrelation"""
    autocorr = librosa.autocorrelate(y)
    return np.max(autocorr)


def extract_spectral_centroid(y, sr):
    """Extract Spectral Centroid"""
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(spectral_centroid)


def extract_spectral_bandwidth(y, sr):
    """Extract Spectral Bandwidth"""
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return np.mean(spectral_bandwidth)


def extract_spectral_rolloff(y, sr):
    """Extract Spectral Roll-off"""
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    return np.mean(spectral_rolloff)


def extract_mfcc(y, sr, n_mfcc=13):
    """Extract MFCCs"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)


def extract_chromagram(y, sr):
    """Extract Chromagram"""
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chromagram, axis=1)


def extract_mel_spectrogram(y, sr, n_mels=128):
    """Extract Mel Spectrogram"""
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return np.mean(mel_spectrogram, axis=1)


def extract_tonnetz(y, sr):
    """Extract Tonnetz"""
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    return np.mean(tonnetz, axis=1)


def extract_spectral_contrast(y, sr, n_bands=4, fmin=50.0):
    """Extract Spectral Contrast

    Args:
        y: Audio signal
        sr: Sampling rate
        n_bands: Number of frequency bands (default: 4)
        fmin: Minimum frequency (default: 50.0)

    Returns:
        float: Mean spectral contrast
    """
    # Adjust parameters based on sampling rate to avoid Nyquist issues
    nyquist = sr / 2
    max_bands = int(np.floor(np.log2(nyquist / fmin)))
    n_bands = min(n_bands, max_bands)

    try:
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=sr,
            n_bands=n_bands,
            fmin=fmin
        )
        return np.mean(spectral_contrast)
    except librosa.util.exceptions.ParameterError:
        # If parameters still cause issues, return a default value or raise custom exception
        return 0.0  # or raise your own exception if preferred


def extract_temporal_dynamics(y, sr, n_mfcc=13):
    """Extract Temporal Dynamics"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    return np.mean(mfcc_delta, axis=1), np.mean(mfcc_delta2, axis=1)


def extract_pitch(y, sr):
    """Extract Pitch"""
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])
    return pitch


def extract_rhythm_tempo(y, sr):
    """Extract Rhythm and Tempo

    Args:
        y: Audio signal
        sr: Sampling rate

    Returns:
        float: Tempo value or 0.0 if unable to detect
    """
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo) if not np.isnan(tempo) else 0.0
    except:
        return 0.0


def extract_beat_frames(y, sr):
    """Extract beat frames

    Args:
        y: Audio signal
        sr: Sampling rate

    Returns:
        np.ndarray: Beat frames
    """
    _, beats = librosa.beat.beat_track(y=y, sr=sr)
    return beats


def extract_formants(y, sr, order=2):
    """Extract Formants using LPC"""
    lpc = librosa.core.lpc(y, order=order)
    roots = np.roots(lpc)
    roots = [r for r in roots if np.imag(r) >= 0]
    angz = np.arctan2(np.imag(roots), np.real(roots))
    formants = sorted(angz * (sr / (2 * np.pi)))
    return formants[:2]  # Returning first two formants


def extract_statistical_perceptual_features(file_path):
    """Load audio and extract statistical and perceptual features"""
    y, sr = librosa.load(file_path)
    features = {
        "zcr": extract_zcr(y),
        "energy": extract_energy(y),
        "rms": extract_rms(y),
        "autocorrelation": extract_autocorrelation(y),
        "spectral_centroid": extract_spectral_centroid(y, sr),
        "spectral_bandwidth": extract_spectral_bandwidth(y, sr),
        "spectral_rolloff": extract_spectral_rolloff(y, sr),
        "spectral_contrast": extract_spectral_contrast(y, sr),
        "pitch": extract_pitch(y, sr),
        "tempo": extract_rhythm_tempo(y, sr)[0],
        "formants": extract_formants(y, sr)
    }
    return features


def extract_advanced_features(file_path):
    """Extract advanced features"""
    y, sr = librosa.load(file_path)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_strength = np.mean(onset_env[beat_frames])

    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    pulse = librosa.beat.plp(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    rhythm_pattern = np.mean(pulse)

    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    onset_rate = len(onsets) / librosa.get_duration(y=y, sr=sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_var = np.var(mfccs, axis=1)

    spec = np.abs(librosa.stft(y))
    spectral_flux = np.mean(librosa.onset.onset_strength(S=librosa.amplitude_to_db(spec, ref=np.max)))

    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

    features = {
        'beat_strength': beat_strength,
        'rhythm_pattern': rhythm_pattern,
        'onset_rate': onset_rate,
        'mfccs_mean': mfccs_mean.tolist(),
        'mfccs_var': mfccs_var.tolist(),
        'spectral_flux': spectral_flux,
        'spectral_flatness': spectral_flatness
    }

    return features


# Regularity Feature Extraction Functions
def extract_tempo(y, sr):
    """Extract Tempo Consistency"""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return np.std(tempo)


def extract_beat_strength(y, sr):
    """Extract Beat Strength Consistency"""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_strength = np.std(onset_env[beat_frames])
    return beat_strength


def extract_onset_rate(y, sr):
    """Extract Onset Rate Consistency"""
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_rate = np.diff(onset_times)
    return np.std(onset_rate)


def extract_spectral_flux(y, sr):
    """Extract Spectral Flux Consistency"""
    spec = np.abs(librosa.stft(y))
    spectral_flux = np.mean(librosa.onset.onset_strength(S=librosa.amplitude_to_db(spec, ref=np.max)))
    return spectral_flux


def extract_audio_features_and_regularity(y, sr):
    """Extract audio features and calculate the regularity metric"""
    features = {
        "tempo_consistency": extract_tempo(y, sr),
        "beat_strength_consistency": extract_beat_strength(y, sr),
        "onset_rate_consistency": extract_onset_rate(y, sr),
        "spectral_flux_consistency": extract_spectral_flux(y, sr)
    }
    return features


def segment_audio(y, sr, segment_duration=5):
    """Segment the audio into first and last parts"""
    first_segment = y[:segment_duration * sr]
    last_segment = y[-segment_duration * sr:]
    return first_segment, last_segment


def extract_regularity_features(file_path):
    """Extract regularity features"""
    y, sr = librosa.load(file_path)

    first_segment, last_segment = segment_audio(y, sr)

    first_segment_features = extract_audio_features_and_regularity(first_segment, sr)
    last_segment_features = extract_audio_features_and_regularity(last_segment, sr)

    features = {
        "tempo_consistency": extract_tempo(y, sr),
        "beat_strength_consistency": extract_beat_strength(y, sr),
        "onset_rate_consistency": extract_onset_rate(y, sr),
        "spectral_flux_consistency": extract_spectral_flux(y, sr),
        "first_segment_tempo_consistency": first_segment_features["tempo_consistency"],
        "first_segment_beat_strength_consistency": first_segment_features["beat_strength_consistency"],
        "first_segment_onset_rate_consistency": first_segment_features["onset_rate_consistency"],
        "first_segment_spectral_flux_consistency": first_segment_features["spectral_flux_consistency"],
        "last_segment_tempo_consistency": last_segment_features["tempo_consistency"],
        "last_segment_beat_strength_consistency": last_segment_features["beat_strength_consistency"],
        "last_segment_onset_rate_consistency": last_segment_features["onset_rate_consistency"],
        "last_segment_spectral_flux_consistency": last_segment_features["spectral_flux_consistency"]
    }
    return features


BASIC_FEATURES = {
    'zcr': extract_zcr,
    'energy': extract_energy,
    'rms': extract_rms,
    'autocorrelation': extract_autocorrelation
}

SPECTRAL_FEATURES = {
    'spectral_centroid': extract_spectral_centroid,
    'spectral_bandwidth': extract_spectral_bandwidth,
    'spectral_rolloff': extract_spectral_rolloff,
    'spectral_contrast': extract_spectral_contrast,
    'spectral_flux': extract_spectral_flux
}

RHYTHM_FEATURES = {
    'tempo': extract_rhythm_tempo,
    'beat_frames': extract_beat_frames,
    'beat_strength': extract_beat_strength,
    'onset_rate': extract_onset_rate
}

# All available feature extractors
AVAILABLE_EXTRACTORS = {
    **BASIC_FEATURES,
    **SPECTRAL_FEATURES,
    **RHYTHM_FEATURES
}