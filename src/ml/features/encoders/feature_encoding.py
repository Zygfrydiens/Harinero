from typing import List, Dict, Union

# Define the categories and their corresponding letters
categories = {
    'time_domain': 't',
    'frequency_domain': 'f',
    'mfcc_mean': 'm',
    'mfcc_var': 'v',
    'spectral_contrast': 'c',
    'consistency': 's',
    'first_segment': 'fs',
    'last_segment': 'ls'
}

# List of features under each category
feature_groups = {
    'time_domain': ['zcr', 'energy', 'rms', 'autocorrelation', 'beat_strength', 'rhythm_pattern', 'onset_rate'],
    'frequency_domain': ['spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'pitch', 'tempo', 'formants', 'spectral_flux', 'spectral_flatness'],
    'mfcc_mean': [f'mfcc_mean_{i}' for i in range(13)],
    'mfcc_var': [f'mfcc_var_{i}' for i in range(13)],
    'spectral_contrast': [f'spectral_contrast_{i}' for i in range(1, 8)],
    'consistency': ['beat_strength_consistency', 'onset_rate_consistency', 'spectral_flux_consistency'],
    'first_segment': ['first_segment_beat_strength_consistency', 'first_segment_onset_rate_consistency', 'first_segment_spectral_flux_consistency'],
    'last_segment': ['last_segment_beat_strength_consistency', 'last_segment_onset_rate_consistency', 'last_segment_spectral_flux_consistency']
}

# Initialize the dictionaries
original_to_encoded = {}
encoded_to_original = {}

# Generate the encoded names
for category, prefix in categories.items():
    features = feature_groups[category]
    for i, feature in enumerate(features, start=1):
        encoded_name = f"{prefix}{i}"
        original_to_encoded[feature] = encoded_name
        encoded_to_original[encoded_name] = feature


def encode_feature_names(feature_names: List[str]) -> List[str]:
    """Encodes feature names to their short form.

    Args:
        feature_names: List of original feature names to encode

    Returns:
        List of encoded feature names

    Example:
        >>> encode_feature_names(['spectral_centroid', 'mfcc_mean_0'])
        ['f1', 'm1']
    """
    return [original_to_encoded.get(name, name) for name in feature_names]


def decode_feature_names(encoded_names: List[str]) -> List[str]:
    """Decodes encoded feature names to their original form.

    Args:
        encoded_names: List of encoded feature names to decode

    Returns:
        List of original feature names

    Example:
        >>> decode_feature_names(['f1', 'm1'])
        ['spectral_centroid', 'mfcc_mean_0']
    """
    return [encoded_to_original.get(name, name) for name in encoded_names]


# Print the dictionaries for verification (optional)
if __name__ == "__main__":
    print("Original to Encoded Mapping:")
    print(original_to_encoded)
    print("\nEncoded to Original Mapping:")
    print(encoded_to_original)
