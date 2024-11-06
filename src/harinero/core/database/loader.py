from typing import Tuple
import sqlite3
import pandas as pd
from pathlib import Path


def load_data_from_db(db_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load core data tables from the database.

    Reason:
        Centralized database loading to ensure consistent data access across the application.
        Returns all core tables needed for song finding and processing.

    Args:
        db_path (Path): Path to the SQLite database file

    Raises:
        sqlite3.Error: If database connection or query fails
        FileNotFoundError: If database file doesn't exist

    Returns:
        Tuple containing DataFrames in order:
            - authors_df: Author information
            - albums_df: Album information
            - songs_df: Song information
            - songs_metadata_df: Song metadata and features
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        authors_df = pd.read_sql_query("SELECT * FROM authors", conn)
        albums_df = pd.read_sql_query("SELECT * FROM albums", conn)
        songs_df = pd.read_sql_query("SELECT * FROM songs", conn)
        songs_metadata_df = pd.read_sql_query("SELECT * FROM songs_metadata", conn)
    finally:
        conn.close()

    return authors_df, albums_df, songs_df, songs_metadata_df