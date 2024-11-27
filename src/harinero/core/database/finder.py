"""Database interface for finding and retrieving tango songs.

TODO (HIGH PRIORITY):
   Refactor to use SongSearchCriteria dataclass for search parameters to improve:
   - Type safety
   - Parameter validation
   - API clarity
   - Maintainability

   Example structure:
   @dataclass
   class SongSearchCriteria:
       author_name: Optional[str] = None
       year: Optional[Union[int, Tuple[int, int]]] = None
       genre: Optional[str] = None
       singer: Optional[str] = None

TODO (MEDIUM PRIORITY):
   - Add logging for better debugging and monitoring
   - Add caching for frequently accessed data
   - Convert string paths to Path objects
"""

from ..models.structures import SongStruct
from .loader import load_data_from_db
import numpy as np
from rapidfuzz import process, fuzz
import unicodedata
import os


class SongFinder:
    """Database interface for finding and retrieving tango songs.

    This class provides methods to search and retrieve songs from the database
    based on various criteria. It maintains connections to all relevant database
    tables and handles the complexity of joining related song information.

    Attributes:
        _authors_df: DataFrame containing author/orchestra information
        _albums_df: DataFrame containing album information
        _songs_df: DataFrame containing song basic information
        _songs_metadata_df: DataFrame containing song features and metadata
        _media_path: Path to the media files directory
    """

    def __init__(self, config):
        """Initialize SongFinder with database connection and configuration.

        Args:
            config: Configuration dictionary containing:
                - paths.database: Path to SQLite database
                - paths.media: Path to media files directory

        Raises:
            ValueError: If database tables are empty
            FileNotFoundError: If database file doesn't exist
        """
        self._authors_df, self._albums_df, self._songs_df, self._songs_metadata_df, self._songs_moods_df = load_data_from_db(
            config['paths']['database'])
        self._media_path = config['paths']['media']
        self._validate_dataframes()

    def _validate_dataframes(self):
        """Validate that all required database tables contain data.

        Raises:
            ValueError: If any required table is empty
        """
        if self._authors_df.empty or self._albums_df.empty or self._songs_df.empty or self._songs_metadata_df.empty:
            raise ValueError("One or more input DataFrames are empty.")

    def get_song(self, song_id):
        """Retrieve complete song information including metadata and related entities.

        This method joins information from songs, albums, authors, and metadata tables
        to create a complete song profile.

        Args:
            song_id: Unique identifier of the song

        Raises:
            ValueError: If song_id doesn't exist or if related data is missing/inconsistent

        Returns:
            SongStruct containing complete song information including:
                - Basic song details (name, genre, year)
                - Related entities (album, author)
                - Audio features (tempo, pitch, moods etc.)
                - File location
        """
        song_row = self._songs_df[self._songs_df['song_id'] == song_id]
        if song_row.empty:
            raise ValueError(f"Song with ID {song_id} not found.")

        song_row = song_row.iloc[0]
        album_row = self._albums_df[self._albums_df['album_id'] == song_row['album_id']]
        if album_row.empty:
            raise ValueError(f"Album with ID {song_row['album_id']} not found.")

        album_row = album_row.iloc[0]
        author_row = self._authors_df[self._authors_df['author_id'] == album_row['author_id']]
        if author_row.empty:
            raise ValueError(f"Author with ID {album_row['author_id']} not found.")

        author_row = author_row.iloc[0]
        metadata_row = self._songs_metadata_df[self._songs_metadata_df['song_id'] == song_id]
        moods_row = self._songs_moods_df[self._songs_moods_df['song_id'] == song_id].iloc[
            0] if not self._songs_moods_df.empty else None

        return SongStruct(
            song_id=song_id,
            name=song_row['name'],
            singer=song_row['singer'],
            genre=song_row['genre'],
            track_number=song_row['track_number'],
            year=song_row['year'],
            album_name=album_row['name'],
            author_name=author_row['name'],
            tempo=round(metadata_row['tempo'].values[0], 2),
            beat_strength=round(metadata_row['beat_strength'].values[0], 2),
            pitch=round(metadata_row['pitch'].values[0], 2),
            brightness=round(metadata_row['spectral_centroid'].values[0], 2),
            energy=round(metadata_row['energy'].values[0], 2),
            popularity=song_row.get('popularity'),
            happy=moods_row['happy'] if moods_row is not None else None,
            sad=moods_row['sad'] if moods_row is not None else None,
            romantic=moods_row['romantic'] if moods_row is not None else None,
            dramatic=moods_row['dramatic'] if moods_row is not None else None,
            file_path=os.path.join(self._media_path,
                                   metadata_row['file_path'].values[0]) if not metadata_row.empty else None
        )

    def get_songs_by_criteria(self, author_name=None, year=None, genre=None, singer=None):
        """Find songs matching specified search criteria.

        Provides flexible song searching based on multiple criteria. All criteria are
        optional - only the provided criteria are used to filter songs.

        Args:
            author_name: Orchestra/author name to filter by
            year: Either specific year (int) or (start_year, end_year) tuple
            genre: Tango genre to filter by (tango, vals, milonga, etc.)
            singer: Singer name to filter by

        Raises:
            ValueError: If any specified criteria value is invalid or not found

        Returns:
            List of SongStruct objects matching all specified criteria
        """
        songs_filtered = self._songs_df.copy()

        if author_name:
            author_name = author_name.lower()
            authors_filtered = self._authors_df[self._authors_df['name'].str.lower() == author_name]
            if authors_filtered.empty:
                raise ValueError(f"Author '{author_name}' not found.")
            author_ids = authors_filtered['author_id'].tolist()
            albums_filtered = self._albums_df[self._albums_df['author_id'].isin(author_ids)]
            album_ids = albums_filtered['album_id'].tolist()
            songs_filtered = songs_filtered[songs_filtered['album_id'].isin(album_ids)]

        if year:
            if isinstance(year, int):
                songs_filtered = songs_filtered[songs_filtered['year'] == year]
            elif isinstance(year, tuple) and len(year) == 2:
                start_year, end_year = year
                songs_filtered = songs_filtered[
                    (songs_filtered['year'] >= start_year) & (songs_filtered['year'] <= end_year)
                    ]
            else:
                raise ValueError("Year must be an integer or a tuple of two integers.")

        if genre:
            genre = genre.lower()
            if not self._songs_df['genre'].str.lower().isin([genre]).any():
                raise ValueError(f"Genre '{genre}' not found.")
            songs_filtered = songs_filtered[songs_filtered['genre'].str.lower() == genre]

        if singer:
            singer = singer.lower()
            if not self._songs_df['singer'].str.lower().isin([singer]).any():
                raise ValueError(f"Singer '{singer}' not found.")
            songs_filtered = songs_filtered[songs_filtered['singer'].str.lower() == singer]

        return [self.get_song(row['song_id']) for _, row in songs_filtered.iterrows()]

    def get_dissimilar_songs(self, selected_song, sample_size=100, year_range=5):
        """Find songs that are dissimilar to the selected song.

        Used by the ML pipeline to find negative examples for training. Returns songs
        that are from the same genre but by different orchestras and singers,
        weighted by year proximity.

        Args:
            selected_song: Reference song to find dissimilar songs for
            sample_size: Number of dissimilar songs to return (default: 100)
            year_range: Year range parameter for weighting - smaller values give
                       stronger preference to songs from similar years (default: 5)

        Returns:
            List of SongStruct objects representing dissimilar songs

        Raises:
            ValueError: If reference song's album is not found
        """
        selected_genre = selected_song.genre.lower()
        selected_singer = selected_song.singer.lower()
        selected_year = selected_song.year

        album_row = self._albums_df[self._albums_df['name'].str.lower() == selected_song.album_name.lower()]
        if album_row.empty:
            raise ValueError(f"Album '{selected_song.album_name}' not found.")
        selected_author_id = album_row.iloc[0]['author_id']

        # Filter songs by genre and exclude same singer
        songs_filtered = self._songs_df[
            (self._songs_df['genre'].str.lower() == selected_genre) &
            (self._songs_df['singer'].str.lower() != selected_singer)
            ]

        # Exclude songs from the same orchestra
        albums_by_selected_author = self._albums_df[self._albums_df['author_id'] == selected_author_id]['album_id']
        filtered_dissimilar_songs = songs_filtered[~songs_filtered['album_id'].isin(albums_by_selected_author)]

        # Weight by year proximity
        year_diff = abs(filtered_dissimilar_songs['year'] - selected_year)
        max_diff = year_diff.max()
        if max_diff > 0:
            filtered_dissimilar_songs['weight'] = np.exp(-year_diff / year_range)
        else:
            filtered_dissimilar_songs['weight'] = 1.0

        sampled_songs = filtered_dissimilar_songs.sample(n=sample_size, weights='weight', replace=False)

        return [self.get_song(row['song_id']) for _, row in sampled_songs.iterrows()]

    def normalize_text(self, text):
        """Normalize text for comparison by removing accents and converting to lowercase.

        Args:
            text: Text to normalize

        Returns:
            Normalized text without accents and in lowercase
        """
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8').lower()

    def get_songs_by_name(self, song_name):
        """Find songs with similar names using fuzzy matching.

        Uses fuzzy string matching to find songs with similar names, accounting for
        minor spelling differences, accents, and word order variations.

        Args:
            song_name: Name of the song to search for

        Returns:
            Tuple containing:
                - List of SongStruct objects for matching songs
                - DataFrame with detailed match information

        Raises:
            ValueError: If no songs match with sufficient similarity
        """
        song_name_normalized = self.normalize_text(song_name)
        self._songs_df['normalized_name'] = self._songs_df['name'].apply(self.normalize_text)
        matches = process.extract(song_name_normalized,
                                  self._songs_df['normalized_name'],
                                  scorer=fuzz.token_sort_ratio,
                                  limit=10)
        threshold = 80
        best_matches = [match for match, score, idx in matches if score >= threshold]

        if not best_matches:
            raise ValueError(f"No songs found matching '{song_name}' with sufficient similarity.")

        matching_songs = self._songs_df[self._songs_df['normalized_name'].isin(best_matches)]

        return [self.get_song(row['song_id']) for _, row in matching_songs.iterrows()], matching_songs