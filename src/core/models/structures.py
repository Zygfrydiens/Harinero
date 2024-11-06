from dataclasses import dataclass
import subprocess
import os
import sys
from IPython.display import Audio, display
from typing import List, Tuple, Optional, Union


@dataclass(frozen=True)
class SongStruct:
    """Represents a tango song with its metadata and audio features.

    Reason:
        Provides a standardized way to store and access song information
        including metadata and extracted audio features.

    Attributes:
        song_id: Unique identifier for the song
        name: Song title
        singer: Artist performing the song
        genre: Musical genre classification
        track_number: Position in the album
        year: Year of recording/release
        album_name: Name of the album
        author_name: Original composer/author
        file_path: Path to the audio file
        tempo: Detected tempo in BPM
        beat_strength: Measure of rhythmic emphasis
        pitch: Average pitch value
        brightness: Measure of spectral brightness
    """
    song_id: int
    name: str
    singer: str
    genre: str
    track_number: int
    year: int
    album_name: str
    author_name: str
    file_path: str
    tempo: float
    beat_strength: float
    pitch: float
    brightness: float

    def __str__(self) -> str:
        """Creates a formatted string representation of the song.

        Returns:
            Formatted string with song details
        """
        return (f"Song: {self.name} (ID: {self.song_id})\n"
                f"Artist: {self.singer}\n"
                f"Genre: {self.genre}\n"
                f"Track Number: {self.track_number}\n"
                f"Year: {self.year}\n"
                f"Album: {self.album_name}\n"
                f"Author: {self.author_name}\n"
                f"Tempo: {self.tempo}\n"
                f"Beat Strength: {self.beat_strength}\n"
                f"Pitch: {self.pitch}\n"
                f"Brightness: {self.brightness}\n")

    def extract(self, year_range: Optional[int] = None) -> Tuple[str, Union[int, Tuple[int, int]], str, str]:
        """Extracts key song metadata with optional year range calculation.

        Args:
            year_range: Optional range of years around the song's year

        Returns:
            Tuple containing (author_name, year_value, genre, singer) where
            year_value is either the original year or a tuple of (min_year, max_year)
        """
        year_value = (self.year - year_range, self.year + year_range) if year_range is not None else self.year
        return self.author_name, year_value, self.genre, self.singer

    def playex(self) -> None:
        """Plays the song using the system's default audio player.

        Raises:
            Exception: If there's an error playing the audio file
        """
        if os.path.exists(self.file_path):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(self.file_path)
                elif os.name == 'posix':  # macOS or Linux
                    if sys.platform == 'darwin':  # macOS
                        subprocess.call(['open', self.file_path])
                    else:  # Linux
                        subprocess.call(['xdg-open', self.file_path])
            except Exception as e:
                print(f"Error playing the song: {e}")
        else:
            print("File path does not exist.")

    def play(self) -> Optional[Audio]:
        """Creates an IPython Audio widget for playback in Jupyter notebooks.

        Returns:
            Audio widget if file exists, None if file not found or error occurs
        """
        if os.path.exists(self.file_path):
            try:
                return Audio(self.file_path)
            except Exception as e:
                print(f"Error playing the song in Jupyter Notebook: {e}")
                return None
        else:
            print("File path does not exist.")
            return None


@dataclass(frozen=True)
class TandaStruct:
    """Represents a tanda (set of related tango songs) with aggregate features.

    Reason:
        Groups related songs together and provides aggregate statistics
        and playback functionality for the entire tanda.

    Attributes:
        tanda_number: Unique identifier for the tanda
        genre: Musical genre of the tanda
        author: Composer/author of the songs
        singer: Primary artist performing the songs
        average_tempo: Mean tempo across all songs
        average_pitch: Mean pitch across all songs
        average_brightness: Mean brightness across all songs
        average_beat_strength: Mean beat strength across all songs
        songs: List of songs in this tanda
    """
    tanda_number: int
    genre: str
    author: str
    singer: str
    average_tempo: float
    average_pitch: float
    average_brightness: float
    average_beat_strength: float
    songs: List[SongStruct]

    def __str__(self) -> str:
        """Creates a formatted string representation of the tanda.

        Returns:
            Formatted string with tanda details and all songs
        """
        song_details = "\n".join([str(song) for song in self.songs])
        return (f"Tanda Number: {self.tanda_number}\n"
                f"Genre: {self.genre}\n"
                f"Author: {self.author}\n"
                f"Singer: {self.singer}\n"
                f"Average Tempo: {self.average_tempo:.2f}\n"
                f"Average Pitch: {self.average_pitch:.2f}\n"
                f"Average Brightness: {self.average_brightness:.2f}\n"
                f"Average Beat Strength: {self.average_beat_strength:.2f}\n"
                f"Songs:\n{song_details}")

    def extract(self) -> List[SongStruct]:
        """Returns ordered list of songs in the tanda.

        Returns:
            List of SongStruct objects in tanda order
        """
        return self.songs

    def playex(self) -> None:
        """Plays all songs in the tanda using system's default audio player.

        Raises:
            Exception: If there's an error playing any of the audio files
        """
        for song in self.songs:
            if os.path.exists(song.file_path):
                try:
                    print(f"Playing {song.name} by {song.singer}")
                    if os.name == 'nt':  # Windows
                        os.startfile(song.file_path)
                    elif os.name == 'posix':  # macOS or Linux
                        if sys.platform == 'darwin':  # macOS
                            subprocess.call(['open', song.file_path])
                        else:  # Linux
                            subprocess.call(['xdg-open', song.file_path])
                except Exception as e:
                    print(f"Error playing the song {song.name}: {e}")
            else:
                print(f"File path does not exist for song {song.name}.")

    def play(self) -> None:
        """Creates IPython Audio widgets for all songs in the tanda."""
        for song in self.songs:
            print(f"Playing {song.name} by {song.singer}")
            display(Audio(song.file_path, autoplay=False))


@dataclass(frozen=True)
class MilongaStruct:
    """Represents a milonga's complete set of tandas.

    Reason:
        Top-level structure that contains all tandas for a milonga event.
        A milonga is a tango dance event consisting of multiple tandas.

    Attributes:
        tandas: Ordered list of tandas that make up the milonga
    """
    tandas: List[TandaStruct]