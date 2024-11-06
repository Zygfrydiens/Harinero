"""
Data processing module for feature extraction and processing in the Harinero tanda creator project.

This module handles feature extraction and DataFrame processing for audio analysis.
"""

from typing import Dict, List, Callable, Any, Set
import pandas as pd
import numpy as np
from tqdm import tqdm

from ..features.extractors.features import (
    extract_statistical_perceptual_features,
    extract_advanced_features,
    extract_regularity_features,
)
# Type aliases
FeatureFunction = Callable[[str], Dict[str, Any]]
FeatureConfig = Dict[str, List[FeatureFunction]]

# Define the feature extraction configuration
FEATURE_CONFIG: FeatureConfig = {
    'basic': [extract_statistical_perceptual_features],
    'advanced': [extract_advanced_features],
    'regularity': [extract_regularity_features]
}


def initialize_feature_columns(df: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    """Initialize DataFrame with NaN columns for all possible features.

    Args:
        df: Input DataFrame containing at least a 'file_path' column
        feature_config: Dictionary mapping feature categories to extraction functions

    Returns:
        DataFrame with additional columns for all possible features initialized to NaN

    Raises:
        ValueError: If DataFrame is empty or doesn't contain 'file_path' column
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    if 'file_path' not in df.columns:
        raise ValueError("DataFrame must contain 'file_path' column")

    feature_set: Set[str] = set()
    sample_file_path = df.iloc[0]['file_path']

    # Extract sample features to determine all possible feature names
    for category, functions in feature_config.items():
        for func in functions:
            try:
                sample_features = func(sample_file_path)
                for key, value in sample_features.items():
                    if isinstance(value, (list, np.ndarray)):
                        feature_set.update([f"{key}_{i}" for i in range(len(value))])
                    else:
                        feature_set.add(key)
            except Exception as e:
                print(f"Warning: Error processing {sample_file_path} for {category}: {str(e)}")

    # Add NaN columns for new features
    for feature in feature_set:
        if feature not in df.columns:
            df[feature] = np.nan

    return df


def extract_and_add_features(df: pd.DataFrame, feature_config: FeatureConfig) -> pd.DataFrame:
    """Extract features from audio files and add them to the DataFrame.

    Args:
        df: Input DataFrame containing 'file_path' column and initialized feature columns
        feature_config: Dictionary mapping feature categories to extraction functions

    Returns:
        DataFrame with extracted feature values

    Raises:
        ValueError: If DataFrame is empty or doesn't contain 'file_path' column
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    if 'file_path' not in df.columns:
        raise ValueError("DataFrame must contain 'file_path' column")

    for category, functions in feature_config.items():
        for func in functions:
            for index, row in tqdm(df.iterrows(),
                                   total=df.shape[0],
                                   desc=f"Extracting {category} features"):
                file_path = row['file_path']
                try:
                    features = func(file_path)
                    for feature, value in features.items():
                        if isinstance(value, (list, np.ndarray)):
                            for i, v in enumerate(value):
                                df.at[index, f'{feature}_{i}'] = v
                        else:
                            df.at[index, feature] = value
                except Exception as e:
                    print(f"Warning: Error processing {file_path}: {str(e)}")
                    continue

    return df


def save_df(df: pd.DataFrame, output_file_path: str) -> None:
    """Save DataFrame to a CSV file.

    Args:
        df: DataFrame to save
        output_file_path: Path where to save the CSV file

    Raises:
        IOError: If unable to save the file
    """
    try:
        df.to_csv(output_file_path, index=False)
        print(f"DataFrame successfully saved to {output_file_path}")
    except IOError as e:
        raise IOError(f"Error saving DataFrame to {output_file_path}: {str(e)}")


def process_song_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Process song metadata by extracting and adding all configured features.

    This is the main function to use for processing a DataFrame of songs. It will:
    1. Initialize all necessary feature columns
    2. Extract and add all features from the configured feature extractors

    Args:
        df: Input DataFrame containing at least a 'file_path' column

    Returns:
        DataFrame with all features extracted and added

    Example:
        >>> df = pd.DataFrame({'file_path': ['song1.mp3', 'song2.mp3']})
        >>> processed_df = process_song_metadata(df)
    """
    df = initialize_feature_columns(df, FEATURE_CONFIG)
    df = extract_and_add_features(df, FEATURE_CONFIG)
    return df