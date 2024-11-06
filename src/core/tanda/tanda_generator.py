from typing import List, Set, Dict, Any
import pandas as pd
from dataclasses import dataclass

from ..models.structures import TandaStruct, MilongaStruct, SongStruct


class TandaGenerator:
    """Generator for creating and managing tandas and milongas.

    Reason:
        Provides core functionality for creating tandas (sets of tango songs)
        and generating milonga structures while ensuring musical consistency
        and proper song grouping.
    """

    def __init__(self) -> None:
        """Initialize the TandaGenerator."""
        self.tandas: List[TandaStruct] = []

    def _validate_tanda_number(self, tanda_number: int) -> None:
        """Validate that tanda number is unique.

        Args:
            tanda_number: The proposed tanda number

        Raises:
            ValueError: If tanda number already exists
        """
        if any(tanda.tanda_number == tanda_number for tanda in self.tandas):
            raise ValueError(f"Tanda number {tanda_number} already exists")

    def _validate_song_consistency(self, songs: List[SongStruct]) -> None:
        """Validate that all songs in a tanda are consistent.

        Args:
            songs: List of songs to validate

        Raises:
            ValueError: If songs don't meet tanda requirements
        """
        if not (3 <= len(songs) <= 4):
            raise ValueError("A Tanda must consist of 3 or 4 songs.")

        # Check unique attributes
        attributes_to_check = {
            'author': {song.author_name for song in songs},
            'genre': {song.genre for song in songs},
            'singer': {song.singer for song in songs}
        }

        for attr_name, unique_values in attributes_to_check.items():
            if len(unique_values) > 1:
                raise ValueError(f"All songs in a Tanda must have the same {attr_name}.")

    def _calculate_averages(self, songs: List[SongStruct]) -> Dict[str, float]:
        """Calculate average musical attributes for a set of songs.

        Args:
            songs: List of songs to analyze

        Returns:
            Dictionary containing averaged musical attributes
        """
        attributes = ['tempo', 'pitch', 'brightness', 'beat_strength']
        return {
            f"average_{attr}": sum(getattr(song, attr) for song in songs) / len(songs)
            for attr in attributes
        }

    def create_tanda(self, songs: List[SongStruct], tanda_number: int) -> TandaStruct:
        """Create a new tanda from a list of songs.

        Args:
            songs: List of songs to include in the tanda
            tanda_number: Sequential number of the tanda in the milonga

        Raises:
            ValueError: If songs don't meet tanda requirements or tanda number exists

        Returns:
            Created TandaStruct object
        """
        self._validate_song_consistency(songs)
        self._validate_tanda_number(tanda_number)

        averages = self._calculate_averages(songs)

        tanda = TandaStruct(
            tanda_number=tanda_number,
            genre=songs[0].genre,
            author=songs[0].author_name,
            singer=songs[0].singer,
            songs=songs,
            **averages
        )
        self.tandas.append(tanda)
        return tanda

    def generate_milonga_overview(self) -> pd.DataFrame:
        """Generate overview DataFrame of all tandas.

        Returns:
            DataFrame containing summary information for each tanda
        """
        return pd.DataFrame([
            {
                'tanda_number': tanda.tanda_number,
                'genre': tanda.genre,
                'author': tanda.author,
                'singer': tanda.singer,
                'average_tempo': tanda.average_tempo,
                'average_pitch': tanda.average_pitch,
                'average_brightness': tanda.average_brightness,
                'average_beat_strength': tanda.average_beat_strength
            }
            for tanda in self.tandas
        ])

    def generate_milonga_detail(self) -> pd.DataFrame:
        """Generate detailed DataFrame of all songs in all tandas.

        Returns:
            DataFrame containing detailed information for each song
        """
        return pd.DataFrame([
            {
                'tanda_number': tanda.tanda_number,
                'genre': tanda.genre,
                'author': tanda.author,
                'singer': tanda.singer,
                'song_name': song.name,
                'tempo': song.tempo,
                'pitch': song.pitch,
                'brightness': song.brightness,
                'beat_strength': song.beat_strength
            }
            for tanda in self.tandas
            for song in tanda.songs
        ])

    def generate_milonga(self) -> MilongaStruct:
        """Generate a milonga structure from all created tandas.

        Returns:
            MilongaStruct object containing all tandas
        """
        return MilongaStruct(tandas=self.tandas)