"""Tests for core data structures (SongStruct, TandaStruct, MilongaStruct).

Tests creation, methods, and behavior of the fundamental data structures
used for representing tango music organization.
"""

import pytest
import dataclasses
from src.core.models.structures import SongStruct, TandaStruct, MilongaStruct
from pathlib import Path


@pytest.fixture
def sample_song() -> SongStruct:
    """Creates a sample song for testing.

    Returns:
        SongStruct with test data
    """
    return SongStruct(
        song_id=1,
        name="La Cumparsita",
        singer="Carlos Gardel",
        genre="Tango",
        track_number=1,
        year=1927,
        album_name="Tango Classics",
        author_name="Gerardo Matos Rodríguez",
        file_path="path/to/song.mp3",
        tempo=120.5,
        beat_strength=0.8,
        pitch=440.0,
        brightness=0.7
    )

@pytest.fixture
def sample_tanda(sample_song) -> TandaStruct:
    """Creates a sample tanda for testing.

    Args:
        sample_song: Song fixture to build tanda with

    Returns:
        TandaStruct with test data
    """
    songs = [sample_song]
    return TandaStruct(
        tanda_number=1,
        genre="Tango",
        author="Gerardo Matos Rodríguez",
        singer="Carlos Gardel",
        average_tempo=120.5,
        average_pitch=440.0,
        average_brightness=0.7,
        average_beat_strength=0.8,
        songs=songs
    )


class TestSongStruct:
    """Test suite for SongStruct class."""

    def test_song_creation(self, sample_song):
        """Test song creation with valid data."""
        assert sample_song.song_id == 1
        assert sample_song.name == "La Cumparsita"
        assert sample_song.singer == "Carlos Gardel"

    def test_song_immutability(self, sample_song):
        """Test that song attributes cannot be modified (frozen=True)."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample_song.name = "New Name"

    def test_extract_without_year_range(self, sample_song):
        """Test extract method without year range."""
        author, year, genre, singer = sample_song.extract()
        assert author == "Gerardo Matos Rodríguez"
        assert year == 1927
        assert genre == "Tango"
        assert singer == "Carlos Gardel"

    def test_extract_with_year_range(self, sample_song):
        """Test extract method with year range."""
        author, year_range, genre, singer = sample_song.extract(year_range=5)
        assert year_range == (1922, 1932)

    def test_str_representation(self, sample_song):
        """Test string representation of song."""
        str_repr = str(sample_song)
        assert "La Cumparsita" in str_repr
        assert "Carlos Gardel" in str_repr
        assert "1927" in str_repr

    def test_play_nonexistent_file(self, sample_song):
        """Test play method with non-existent file."""
        result = sample_song.play()
        assert result is None


class TestTandaStruct:
    """Test suite for TandaStruct class."""

    def test_tanda_creation(self, sample_tanda):
        """Test tanda creation with valid data."""
        assert sample_tanda.tanda_number == 1
        assert sample_tanda.genre == "Tango"
        assert len(sample_tanda.songs) == 1

    def test_tanda_immutability(self, sample_tanda):
        """Test that tanda attributes cannot be modified."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample_tanda.genre = "Vals"

    def test_extract_songs(self, sample_tanda, sample_song):
        """Test extract method returns correct song list."""
        songs = sample_tanda.extract()
        assert len(songs) == 1
        assert songs[0] == sample_song

    def test_str_representation(self, sample_tanda):
        """Test string representation of tanda."""
        str_repr = str(sample_tanda)
        assert "Tanda Number: 1" in str_repr
        assert "Genre: Tango" in str_repr
        assert "Carlos Gardel" in str_repr


class TestMilongaStruct:
    """Test suite for MilongaStruct class."""

    def test_milonga_creation(self, sample_tanda):
        """Test milonga creation with valid data."""
        milonga = MilongaStruct(tandas=[sample_tanda])
        assert len(milonga.tandas) == 1
        assert milonga.tandas[0] == sample_tanda

    def test_milonga_immutability(self, sample_tanda):
        """Test that milonga attributes cannot be modified."""
        milonga = MilongaStruct(tandas=[sample_tanda])
        with pytest.raises(dataclasses.FrozenInstanceError):
            milonga.tandas = []