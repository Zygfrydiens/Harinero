import pandas as pd
import sqlite3
from pathlib import Path


def create_test_database(source_db_path: str, target_db_path: str) -> None:
    """
    Creates a minimal test database from the source database.

    Args:
        source_db_path: Path to the source database
        target_db_path: Where to save the test database
    """
    # Create test data directory if it doesn't exist
    Path(target_db_path).parent.mkdir(parents=True, exist_ok=True)

    # Connect to source database
    source_conn = sqlite3.connect(source_db_path)

    # Read minimal data from each table
    authors_df = pd.read_sql("SELECT * FROM authors LIMIT 2", source_conn)
    albums_df = pd.read_sql(
        f"SELECT * FROM albums WHERE author_id IN ({','.join(map(str, authors_df['author_id']))})",
        source_conn
    )
    songs_df = pd.read_sql(
        f"SELECT * FROM songs WHERE album_id IN ({','.join(map(str, albums_df['album_id']))})",
        source_conn
    )
    metadata_df = pd.read_sql(
        f"SELECT * FROM songs_metadata WHERE song_id IN ({','.join(map(str, songs_df['song_id']))})",
        source_conn
    )

    # Create test database
    target_conn = sqlite3.connect(target_db_path)

    # Save to test database
    authors_df.to_sql('authors', target_conn, index=False, if_exists='replace')
    albums_df.to_sql('albums', target_conn, index=False, if_exists='replace')
    songs_df.to_sql('songs', target_conn, index=False, if_exists='replace')
    metadata_df.to_sql('songs_metadata', target_conn, index=False, if_exists='replace')

    source_conn.close()
    target_conn.close()


# Usage example:
if __name__ == "__main__":
    create_test_database(
        source_db_path=r"C:\Users\Admin\Documents\Harinero_Project\data\db\HarineroTandaDB.db",
        target_db_path="../data/test.db"
    )