"""Tests for the TandaGenerator class.

This module provides comprehensive testing for tanda creation and validation,
using fixtures for flexible test data generation.
"""

import pytest
from typing import List, Callable, Dict, Any
from pathlib import Path
from src.harinero.core.models.structures import TandaStruct, SongStruct, MilongaStruct
from src.harinero.core.tanda.tanda_generator import TandaGenerator

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """Provide base test data configuration.

    Returns:
        Dictionary containing base test values
    """
    return {
        'author': "D'Arienzo",
        'singer': "Echagüe",
        'genre': "tango",
        'year': 1941,
        'album': "D'Arienzo 1941",
        'base_path': Path("/path/to/songs")
    }

@pytest.fixture
def create_song(test_data: Dict[str, Any]) -> Callable[..., SongStruct]:
    """Factory fixture for creating test songs with custom attributes.

    Args:
        test_data: Base test configuration

    Returns:
        Function that creates SongStruct instances with custom attributes
    """
    def _create_song(**kwargs) -> SongStruct:
        song_id = kwargs.get('song_id', 1)
        defaults = {
            'song_id': song_id,
            'name': f"Test Song {song_id}",
            'author_name': test_data['author'],
            'genre': test_data['genre'],
            'singer': test_data['singer'],
            'tempo': 120.0,
            'pitch': 220.0,
            'brightness': 0.7,
            'beat_strength': 0.8,
            'track_number': song_id,
            'year': test_data['year'],
            'album_name': test_data['album'],
            'file_path': str(test_data['base_path'] / f"song_{song_id}.mp3")
        }
        defaults.update(kwargs)
        return SongStruct(**defaults)

    return _create_song

@pytest.fixture
def sample_songs(create_song: Callable[..., SongStruct]) -> List[SongStruct]:
    """Create a standard set of test songs.

    Args:
        create_song: Factory function for creating songs

    Returns:
        List of three consistent test songs
    """
    return [
        create_song(
            song_id=i,
            tempo=120.0 + i,
            pitch=220.0 + i,
            brightness=0.7 + i * 0.01,
            beat_strength=0.8 + i * 0.01
        )
        for i in range(1, 4)
    ]

@pytest.fixture
def tanda_generator() -> TandaGenerator:
    """Create a fresh TandaGenerator instance.

    Returns:
        Clean TandaGenerator instance
    """
    return TandaGenerator()

class TestTandaGenerator:
    """Test suite for TandaGenerator class."""

    def test_initialization(self, tanda_generator: TandaGenerator) -> None:
        """Test TandaGenerator initializes with empty tandas list."""
        assert len(tanda_generator.tandas) == 0

    def test_create_valid_tanda(
            self,
            tanda_generator,
            sample_songs: List[SongStruct],
            test_data: Dict[str, Any]
    ) -> None:
        """Test creating a valid tanda with consistent songs."""
        tanda = tanda_generator.create_tanda(sample_songs, tanda_number=1)

        # Test basic tanda properties
        assert isinstance(tanda, TandaStruct)
        assert tanda.tanda_number == 1
        assert tanda.author == test_data['author']
        assert tanda.genre == test_data['genre']
        assert tanda.singer == test_data['singer']
        assert len(tanda.songs) == 3

        # Test musical averages
        assert tanda.average_tempo == pytest.approx(121.0, rel=1e-2)
        assert tanda.average_pitch == pytest.approx(221.0, rel=1e-2)
        assert tanda.average_brightness == pytest.approx(0.72, rel=1e-2)
        assert tanda.average_beat_strength == pytest.approx(0.82, rel=1e-2)

    def test_invalid_tanda_size(
        self,
        tanda_generator: TandaGenerator,
        create_song: Callable[..., SongStruct]
    ) -> None:
        """Test tanda creation with invalid number of songs."""
        # Test with too few songs
        too_few = [create_song(song_id=i) for i in range(1, 3)]
        with pytest.raises(ValueError, match="must consist of 3 or 4 songs"):
            tanda_generator.create_tanda(too_few, tanda_number=1)

        # Test with too many songs
        too_many = [create_song(song_id=i) for i in range(1, 6)]
        with pytest.raises(ValueError, match="must consist of 3 or 4 songs"):
            tanda_generator.create_tanda(too_many, tanda_number=1)

    @pytest.mark.parametrize("field,value,error_pattern", [
        ("author_name", "Pugliese", "same author"),
        ("genre", "vals", "same genre"),
        ("singer", "Morán", "same singer"),
    ])
    def test_inconsistent_songs(
        self,
        tanda_generator: TandaGenerator,
        create_song: Callable[..., SongStruct],
        field: str,
        value: str,
        error_pattern: str
    ) -> None:
        """Test tanda creation with inconsistent song attributes."""
        songs = [create_song(song_id=i) for i in range(1, 4)]
        # Modify middle song to create inconsistency
        songs[1] = create_song(song_id=2, **{field: value})

        with pytest.raises(ValueError, match=error_pattern):
            tanda_generator.create_tanda(songs, tanda_number=1)

    def test_generate_milonga_overview(
        self,
        tanda_generator: TandaGenerator,
        sample_songs: List[SongStruct]
    ) -> None:
        """Test generation of milonga overview DataFrame."""
        tanda_generator.create_tanda(sample_songs, tanda_number=1)
        overview_df = tanda_generator.generate_milonga_overview()

        assert len(overview_df) == 1
        expected_columns = {
            'tanda_number', 'genre', 'author', 'singer',
            'average_tempo', 'average_pitch', 'average_brightness',
            'average_beat_strength'
        }
        assert set(overview_df.columns) == expected_columns

        # Verify data correctness
        row = overview_df.iloc[0]
        assert row['tanda_number'] == 1
        assert row['genre'] == "tango"
        assert row['author'] == "D'Arienzo"
        assert row['average_tempo'] == pytest.approx(121.0, rel=1e-2)

    def test_generate_milonga_detail(
        self,
        tanda_generator: TandaGenerator,
        sample_songs: List[SongStruct]
    ) -> None:
        """Test generation of detailed milonga DataFrame."""
        tanda_generator.create_tanda(sample_songs, tanda_number=1)
        detail_df = tanda_generator.generate_milonga_detail()

        assert len(detail_df) == 3  # One row per song
        expected_columns = {
            'tanda_number', 'genre', 'author', 'singer', 'song_name',
            'tempo', 'pitch', 'brightness', 'beat_strength'
        }
        assert set(detail_df.columns) == expected_columns

        # Verify each song is present
        assert set(detail_df['song_name']) == {
            "Test Song 1",
            "Test Song 2",
            "Test Song 3"
        }

    def test_multiple_tandas(
        self,
        tanda_generator: TandaGenerator,
        create_song: Callable[..., SongStruct]
    ) -> None:
        """Test creating multiple tandas and generating milonga."""
        # Create two tandas with different songs
        tanda1_songs = [create_song(song_id=i) for i in range(1, 4)]
        tanda2_songs = [create_song(song_id=i) for i in range(4, 7)]

        tanda_generator.create_tanda(tanda1_songs, tanda_number=1)
        tanda_generator.create_tanda(tanda2_songs, tanda_number=2)

        milonga = tanda_generator.generate_milonga()
        assert isinstance(milonga, MilongaStruct)
        assert len(milonga.tandas) == 2

        # Test overview
        overview_df = tanda_generator.generate_milonga_overview()
        assert len(overview_df) == 2
        assert set(overview_df['tanda_number']) == {1, 2}