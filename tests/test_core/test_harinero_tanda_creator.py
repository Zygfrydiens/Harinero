from typing import Dict, Any
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path

from src.core.tanda.harinero_tanda_creator import HarineroTandaCreator
from src.core.models.structures import SongStruct


class TestHarineroTandaCreator:
    """Tests for HarineroTandaCreator class.

    Reason:
        Ensures the reliable functioning of the tanda creation system
        by testing each component thoroughly.
    """

    @pytest.fixture
    def mock_song(self) -> SongStruct:
        """Creates a mock song for testing.

        Returns:
            SongStruct: A fully populated test song instance
        """
        return SongStruct(
            song_id=1,
            name="Test Tango",
            author_name="Test Author",
            year=1940,
            genre="tango",
            singer="Test Singer",
            tempo=120,
            beat_strength=0.8,
            pitch=220,
            brightness=0.7,
            track_number=1,
            album_name="Test Album",
            file_path="test/path/to/audio.mp3"
        )

    @pytest.fixture
    def mock_similar_song(self) -> SongStruct:
        """Creates a similar mock song for testing.

        Returns:
            SongStruct: A similar test song with slightly different values
        """
        return SongStruct(
            song_id=2,
            name="Similar Test Tango",
            author_name="Test Author",  # Same author
            year=1941,  # Close year
            genre="tango",  # Same genre
            singer="Test Singer",  # Same singer
            tempo=122,  # Similar tempo
            beat_strength=0.78,
            pitch=225,
            brightness=0.72,
            track_number=2,
            album_name="Another Test Album",
            file_path="test/path/to/audio2.mp3"
        )

    @pytest.fixture
    def mocked_dependencies(self):
        """Creates mocked dependencies for testing.

        Returns:
            Dict containing mocked instances of all dependencies
        """
        with patch('src.core.tanda.harinero_tanda_creator.SongFinder') as mock_finder, \
                patch('src.core.tanda.harinero_tanda_creator.Preprocessor') as mock_preprocessor, \
                patch('src.core.tanda.harinero_tanda_creator.KerasSongModel') as mock_model, \
                patch('src.core.tanda.harinero_tanda_creator.TandaGenerator') as mock_generator:
            # Configure mocks with return values
            mock_finder_instance = MagicMock()
            mock_preprocessor_instance = MagicMock()
            mock_model_instance = MagicMock()
            mock_generator_instance = MagicMock()

            mock_finder.return_value = mock_finder_instance
            mock_preprocessor.return_value = mock_preprocessor_instance
            mock_model.return_value = mock_model_instance
            mock_generator.return_value = mock_generator_instance

            yield {
                'finder': mock_finder_instance,
                'preprocessor': mock_preprocessor_instance,
                'model': mock_model_instance,
                'generator': mock_generator_instance
            }

    @pytest.fixture
    def creator(self, test_config: Dict[str, Any], mocked_dependencies) -> HarineroTandaCreator:
        """Creates HarineroTandaCreator instance with test configuration."""
        return HarineroTandaCreator(test_config)

    def test_initialization(self, creator: HarineroTandaCreator, test_config):
        """Test proper initialization of HarineroTandaCreator."""
        assert creator.config == test_config
        assert creator.song_finder is not None
        assert creator.preprocessor is not None
        assert creator.model is not None
        assert creator.tanda_generator is not None

    def test_train_on_song(self, creator: HarineroTandaCreator, mock_song: SongStruct):
        """Test training on a single song."""
        # Setup mocks
        creator.song_finder.get_song.return_value = mock_song
        creator.song_finder.get_dissimilar_songs.return_value = [mock_song]
        creator.preprocessor.prepare_dataset_from_song_structs.return_value = (
            np.array([[1.0, 2.0], [3.0, 4.0]]),  # frames
            np.array([1, 0])  # labels
        )
        creator.model.train_on_data.return_value = ({}, {"accuracy": 0.9})

        # Execute
        history, results = creator.train_on_song(1)

        # Verify
        assert results["accuracy"] == 0.9
        creator.song_finder.get_song.assert_called_once_with(1)
        creator.preprocessor.prepare_dataset_from_song_structs.assert_called_once()
        creator.model.train_on_data.assert_called_once()

    def test_predict_on_song(self, creator: HarineroTandaCreator, mock_song: SongStruct):
        """Test prediction on a single song."""
        # Setup
        creator.song_finder.get_song.return_value = mock_song
        creator.preprocessor.slice_audio_from_song_struct.return_value = (
            np.array([[1.0]]),  # frames
            None  # labels not needed for prediction
        )
        creator.model.predict_on_data.return_value = 0.85

        # Execute
        result = creator.predict_on_song(1)

        # Verify
        assert result == 0.85
        creator.song_finder.get_song.assert_called_once_with(1)
        creator.preprocessor.slice_audio_from_song_struct.assert_called_once()
        creator.model.predict_on_data.assert_called_once()

    def test_get_similar_songs_with_probabilities(
            self,
            creator: HarineroTandaCreator,
            mock_song: SongStruct,
            mock_similar_song: SongStruct
    ):
        """Test finding similar songs with probability scores."""
        # Setup
        creator.song_finder.get_song.return_value = mock_song
        creator.song_finder.get_songs_by_criteria.return_value = [mock_song, mock_similar_song]

        # Mock the predict_on_song method
        with patch.object(creator, 'predict_on_song', side_effect=[0.95, 0.85]):
            # Execute
            result_df = creator.get_similar_songs_with_probabilities(1, year_span=5)

            # Verify structure
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == 2
            expected_columns = {
                'song_id', 'name', 'author', 'year', 'genre', 'singer',
                'tempo', 'beat_strength', 'pitch', 'brightness', 'average_probability'
            }
            assert set(result_df.columns) == expected_columns

            # Verify values
            assert result_df.iloc[0]['average_probability'] == 0.95
            assert result_df.iloc[1]['average_probability'] == 0.85
            assert result_df.iloc[0]['song_id'] == mock_song.song_id

    def test_get_similar_songs_with_probabilities_error_handling(
            self,
            creator: HarineroTandaCreator,
            mock_song: SongStruct
    ):
        """Test error handling in similar songs search."""
        # Setup
        creator.song_finder.get_song.return_value = mock_song
        creator.song_finder.get_songs_by_criteria.side_effect = Exception("Database error")

        # Execute and verify
        with pytest.raises(Exception, match="Database error"):
            creator.get_similar_songs_with_probabilities(1, year_span=5)

    def test_song_extract_with_year_span(self, mock_song: SongStruct):
        """Test the extract method of SongStruct with year span."""
        # Test with year span
        year_span = 5
        author, year_range, genre, singer = mock_song.extract(year_span)

        assert author == mock_song.author_name
        assert year_range == (mock_song.year - year_span, mock_song.year + year_span)
        assert genre == mock_song.genre
        assert singer == mock_song.singer

        # Test without year span
        author, year, genre, singer = mock_song.extract()
        assert year == mock_song.year  # Should return single year when no span provided