from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from ..database.finder import SongFinder
from ..models.structures import SongStruct, TandaStruct
from .tanda_generator import TandaGenerator

from ...ml.preprocessing.preprocessor import Preprocessor
from ...ml.models.model_keras import KerasSongModel
from ...utils import load_config

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path


class HarineroTandaCreator:
    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        """Initialize the HarineroTandaCreator.

        Args:
            config: Either a path to config file or config dictionary

        Raises:
            TypeError: If config is neither a string/Path nor a dictionary
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError("Config must be either a path or a dictionary")

        self.song_finder = SongFinder(self.config)
        self.preprocessor = Preprocessor(self.config)
        self.model = KerasSongModel(self.config)
        self.tanda_generator = TandaGenerator()

    def train_on_song(self, song_id):
        selected_song = self.song_finder.get_song(song_id)
        all_frames, all_labels = self.preprocessor.prepare_dataset_from_song_structs(
            selected_song,
            self.song_finder.get_dissimilar_songs(
                selected_song,
                sample_size=self.config['data']['dissimilar_sample_size'],
                year_range=self.config['data']['dissimilar_year_range']
            )
        )
        history, results = self.model.train_on_data(all_frames, all_labels)
        return history, results

    def predict_on_song(self, song_id):
        selected_song = self.song_finder.get_song(song_id)
        frames, _ = self.preprocessor.slice_audio_from_song_struct(selected_song)
        return self.model.predict_on_data(frames)

    def create_tanda(self, song_ids, tanda_number):
        songs = [self.song_finder.get_song(song_id) for song_id in song_ids]
        tanda = self.tanda_generator.create_tanda(songs, tanda_number=tanda_number)
        return tanda

    def generate_tanda_overview(self):
        overview_df = self.tanda_generator.generate_milonga_overview()
        return overview_df

    def generate_tanda_detail(self):
        detail_df = self.tanda_generator.generate_milonga_detail()
        return detail_df

    def get_similar_songs_with_probabilities(self, song_id, year_span):
        selected_song = self.song_finder.get_song(song_id)
        author, year, genre, singer = selected_song.extract(year_span)
        similar_songs = self.song_finder.get_songs_by_criteria(author, year, genre, singer)

        song_details = []
        for song in similar_songs:
            average_probability = self.predict_on_song(song.song_id)
            song_details.append({
                'song_id': song.song_id,
                'name': song.name,
                'author': song.author_name,
                'year': song.year,
                'genre': song.genre,
                'singer': song.singer,
                'tempo': song.tempo,
                'beat_strength': song.beat_strength,
                'pitch': song.pitch,
                'brightness': song.brightness,
                'average_probability': average_probability
            })

        df = pd.DataFrame(song_details)
        df_sorted = df.sort_values(by='average_probability', ascending=False)
        return df_sorted