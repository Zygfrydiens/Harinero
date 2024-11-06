"""
Audio preprocessing module for the Harinero tanda creator project.

This module handles audio file loading and slicing operations for preparing
training datasets.
"""

from typing import Dict, List, Tuple
import librosa
import numpy as np
from ...core.models.structures import SongStruct

class Preprocessor:
    """Audio preprocessing class for handling audio files and creating training datasets."""

    def __init__(self, config: Dict[str, Dict[str, float]]):
        """Initialize the preprocessor with configuration.

        Args:
            config: Configuration dictionary containing audio processing parameters
                   Expected format:
                   {
                       'data': {
                           'sampling_rate': float,
                           'frame_time': float,
                           'selected_hop_time': float,
                           'dissimilar_hop_time': float
                       }
                   }
        """
        self.config = config
        self.sampling_rate = config['data']['sampling_rate']
        self.frame_time = config['data']['frame_time']
        self.selected_hop_time = config['data']['selected_hop_time']
        self.dissimilar_hop_time = config['data']['dissimilar_hop_time']

        self.frame_size = int(self.sampling_rate * self.frame_time)
        self.hop_size_selected = int(self.sampling_rate * self.selected_hop_time)
        self.hop_size_dissimilar = int(self.sampling_rate * self.dissimilar_hop_time)

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file at given sample rate.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(file_path, sr=self.sampling_rate)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {file_path}: {str(e)}")

    def slice_song(self, file_path: str, frame_size: int,
                   hop_size: int, label: int) -> Tuple[np.ndarray, np.ndarray]:
        """Slice a song into frames with given parameters.

        Args:
            file_path: Path to the audio file
            frame_size: Size of each frame
            hop_size: Number of samples between frames
            label: Label to assign to all frames

        Returns:
            Tuple of (frames, labels) where:
                frames: np.ndarray of shape (n_frames, frame_size)
                labels: np.ndarray of shape (n_frames,) containing the label
        """
        audio, _ = self.load_audio(file_path)
        frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_size).T
        labels = np.full(len(frames), label)
        return frames, labels

    def slice_dissimilar_songs(self, file_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Slice multiple dissimilar songs into frames.

        Args:
            file_paths: List of paths to audio files

        Returns:
            Tuple of (frames, labels) where:
                frames: np.ndarray containing all frames from all songs
                labels: np.ndarray containing labels (0) for all frames
        """
        frames = []
        labels = []

        for file_path in file_paths:
            audio_frames, audio_labels = self.slice_song(
                file_path, self.frame_size, self.hop_size_dissimilar, label=0
            )
            frames.extend(audio_frames)
            labels.extend(audio_labels)

        return np.array(frames), np.array(labels)

    def slice_audio_from_song_struct(self, song_struct: SongStruct) -> Tuple[np.ndarray, np.ndarray]:
        """Slice a song from a SongStruct into frames.

        Args:
            song_struct: SongStruct containing song information

        Returns:
            Tuple of (frames, labels) where:
                frames: np.ndarray containing all frames from the song
                labels: np.ndarray containing labels (1) for all frames
        """
        return self.slice_song(song_struct.file_path, self.frame_size, self.hop_size_selected, label=1)

    def prepare_dataset_from_song_structs(
            self,
            selected_song: SongStruct,
            dissimilar_songs: List[SongStruct]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare a dataset from selected song and dissimilar songs.

        This method creates a dataset by combining frames from a selected song
        (labeled as 1) and frames from dissimilar songs (labeled as 0).

        Args:
            selected_song: SongStruct for the main song
            dissimilar_songs: List of SongStruct for dissimilar songs

        Returns:
            Tuple of (all_frames, all_labels) where:
                all_frames: np.ndarray containing frames from all songs
                all_labels: np.ndarray containing binary labels (0 or 1)
        """
        similar_frames, similar_labels = self.slice_audio_from_song_struct(selected_song)

        dissimilar_frames, dissimilar_labels = self.slice_dissimilar_songs(
            [song.file_path for song in dissimilar_songs]
        )

        all_frames = np.concatenate((similar_frames, dissimilar_frames), axis=0)
        all_labels = np.concatenate((similar_labels, dissimilar_labels), axis=0)

        return all_frames, all_labels
