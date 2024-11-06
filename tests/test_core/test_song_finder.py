import pytest
import sqlite3
import os
from typing import Dict, Any

# Import the SongFinder class
from src.core.database.finder import SongFinder


def create_empty_test_db(db_path):
    """Creates an empty test database with the correct schema."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create empty tables with the correct schema
    cursor.execute('''CREATE TABLE IF NOT EXISTS authors
                     (author_id INTEGER PRIMARY KEY, name TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS albums
                     (album_id INTEGER PRIMARY KEY, author_id INTEGER, name TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS songs
                     (song_id INTEGER PRIMARY KEY, album_id INTEGER, name TEXT, 
                      singer TEXT, genre TEXT, track_number TEXT, year REAL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS songs_metadata
                     (metadata_id INTEGER PRIMARY KEY, song_id INTEGER, 
                      file_path TEXT, tempo REAL, beat_strength REAL, 
                      pitch REAL, brightness REAL)''')

    conn.commit()
    conn.close()


@pytest.fixture
def song_finder(test_config: Dict[str, Any]):
    """
    Creates a SongFinder instance for testing.

    Args:
        test_config: The test configuration fixture from conftest.py

    Returns:
        SongFinder instance
    """
    return SongFinder(test_config)


def test_song_finder_initialization(song_finder):
    """
    Verifies that SongFinder initializes correctly and loads data.
    """
    assert song_finder is not None
    assert not song_finder._authors_df.empty, "Authors DataFrame should not be empty"
    assert not song_finder._albums_df.empty, "Albums DataFrame should not be empty"
    assert not song_finder._songs_df.empty, "Songs DataFrame should not be empty"
    assert not song_finder._songs_metadata_df.empty, "Songs metadata DataFrame should not be empty"


def test_get_song_exists(song_finder):
    """
    Tests retrieving an existing song.
    """
    # Assuming song_id 1 exists in your test database
    song = song_finder.get_song(1)
    assert song is not None
    assert song.song_id == 1
    assert isinstance(song.name, str)
    assert isinstance(song.genre, str)
    assert isinstance(song.year, (int, float))


def test_get_song_not_exists(song_finder):
    """
    Tests attempting to retrieve a non-existent song.
    """
    with pytest.raises(ValueError) as exc_info:
        song_finder.get_song(99999)  # Using an ID that shouldn't exist
    assert "not found" in str(exc_info.value)


def test_get_songs_by_author(song_finder):
    """Tests retrieving songs by author."""
    author_name = song_finder._authors_df['name'].iloc[0]
    songs = song_finder.get_songs_by_criteria(author_name=author_name)
    assert len(songs) > 0
    for song in songs:
        assert isinstance(song.name, str)
        assert song.author_name == author_name


def test_get_songs_by_genre(song_finder):
    """Tests retrieving songs by genre."""
    # Get first non-null genre
    known_genre = song_finder._songs_df['genre'].dropna().iloc[0]
    songs = song_finder.get_songs_by_criteria(genre=known_genre)
    assert len(songs) > 0
    for song in songs:
        assert isinstance(song.genre, str)
        assert song.genre.lower() == known_genre.lower()


def test_get_songs_by_year(song_finder):
    """Tests retrieving songs by year."""
    # Get a year that exists in the database
    test_year = int(song_finder._songs_df['year'].dropna().iloc[0])
    songs = song_finder.get_songs_by_criteria(year=test_year)
    assert len(songs) > 0
    for song in songs:
        assert song.year == test_year


def test_get_songs_by_invalid_criteria(song_finder):
    """
    Tests retrieving songs with invalid criteria.
    """
    with pytest.raises(ValueError):
        song_finder.get_songs_by_criteria(author_name="Non Existent Author")


def test_normalize_text(song_finder):
    """Tests text normalization method."""
    test_cases = [
        ("áéíóú", "aeiou"),  # Accents
        ("Ñ", "n"),          # Spanish letters
        ("UPPER", "upper"),  # Uppercase
        ("Mix Case", "mix case"),  # Mixed case
    ]

    for input_text, expected in test_cases:
        result = song_finder.normalize_text(input_text)
        assert result == expected, f"Failed for input '{input_text}': expected '{expected}', got '{result}'"


def test_get_songs_by_name(song_finder):
    """Tests fuzzy song name search."""
    # Get a known song name from test database
    known_song = song_finder._songs_df['name'].iloc[0]

    # Test exact match
    songs, matches = song_finder.get_songs_by_name(known_song)
    assert len(songs) > 0

    # Test partial match (use more characters)
    partial_name = known_song[:int(len(known_song) * 0.7)]  # Use 70% of the name
    songs, matches = song_finder.get_songs_by_name(partial_name)
    assert len(songs) > 0


def test_get_dissimilar_songs(song_finder):
    """Tests dissimilar songs retrieval."""
    # Get a reference song
    reference_song = song_finder.get_song(song_finder._songs_df['song_id'].iloc[0])

    # Get dissimilar songs
    dissimilar_songs = song_finder.get_dissimilar_songs(
        reference_song,
        sample_size=5,
        year_range=10
    )

    assert len(dissimilar_songs) > 0

    # Verify dissimilarity criteria
    for song in dissimilar_songs:
        # Different singer
        assert song.singer.lower() != reference_song.singer.lower()
        # Same genre
        assert song.genre == reference_song.genre
        # Not from same author's album
        assert song.author_name != reference_song.author_name


def test_get_songs_by_year_range(song_finder):
    """Tests getting songs within a year range."""
    # Get min and max years from test database
    min_year = int(song_finder._songs_df['year'].min())
    max_year = int(song_finder._songs_df['year'].max())
    mid_year = (min_year + max_year) // 2

    # Test year range
    songs = song_finder.get_songs_by_criteria(year=(mid_year - 1, mid_year + 1))

    assert len(songs) > 0
    for song in songs:
        assert mid_year - 1 <= song.year <= mid_year + 1


def test_error_handling(song_finder):
    """Tests various error conditions."""

    # Invalid song ID
    with pytest.raises(ValueError, match="not found"):
        song_finder.get_song(-1)

    # Invalid author
    with pytest.raises(ValueError, match="not found"):
        song_finder.get_songs_by_criteria(author_name="NonExistent Author")

    # Invalid year format
    with pytest.raises(ValueError, match="must be an integer or a tuple"):
        song_finder.get_songs_by_criteria(year="1940")  # String instead of int

    # Invalid year range
    with pytest.raises(ValueError, match="must be an integer or a tuple"):
        song_finder.get_songs_by_criteria(year=(1940,))  # Incomplete tuple


def test_song_finder_with_empty_tables(test_config):
    """Tests handling of empty database tables."""
    # Create empty test database
    empty_db_path = os.path.join(
        os.path.dirname(test_config['paths']['database']), 'empty_test.db'
    )
    create_empty_test_db(empty_db_path)

    with pytest.raises(ValueError, match="One or more input DataFrames are empty"):
        empty_config = test_config.copy()
        empty_config['paths']['database'] = empty_db_path
        SongFinder(empty_config)


def test_song_finder_invalid_database_path(test_config):
    """Tests handling of invalid database path.

    Verifies that attempting to initialize SongFinder with a non-existent
    database path raises the appropriate error.
    """
    bad_config = test_config.copy()
    bad_config['paths']['database'] = 'nonexistent/path/db.db'

    with pytest.raises(FileNotFoundError, match="Database file not found"):
        SongFinder(bad_config)


def test_get_song_with_missing_metadata(song_finder):
    """Tests handling of songs with missing metadata."""
    # Get a song ID that exists in songs table but not in metadata
    song_id = song_finder._songs_df['song_id'].iloc[0]
    song = song_finder.get_song(song_id)

    # Even with missing metadata, core song info should be present
    assert song.song_id == song_id
    assert isinstance(song.name, str)
    # Metadata fields can be None but shouldn't raise errors
    assert hasattr(song, 'tempo')