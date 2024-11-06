from pydub import AudioSegment
import os
import json
from datetime import datetime
import shutil
import tkinter as tk


class PhraseDataset:
    def __init__(self, base_path):
        """Initialize dataset handler."""
        self.base_path = base_path
        self.metadata_path = os.path.join(base_path, "metadata.json")

        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)

        # Load or initialize metadata
        self.metadata = self._load_metadata()

        # Reference to GUI root for messagebox (will be set by GUI)
        self.root = None

    def set_root(self, root):
        """Set the root window for dialogs."""
        self.root = root

    def extract_segment(self, source_path, start_time, end_time):
        """Extract segment from audio file.

        Args:
            source_path: Path to source audio file
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            AudioSegment object containing the extracted segment
        """
        try:
            # Load audio file
            audio = AudioSegment.from_mp3(source_path)

            # Convert times to milliseconds
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)

            # Extract segment
            segment = audio[start_ms:end_ms]

            return segment

        except Exception as e:
            raise Exception(f"Error extracting segment: {str(e)}")

    def _get_song_folder_name(self, song_id, file_path):
        """Create standardized folder name from song details."""
        # Extract the original filename without extension
        original_name = os.path.splitext(os.path.basename(file_path))[0]
        # Create folder name
        folder_name = f"{song_id}_{original_name}"
        return folder_name

    def _ensure_song_folder(self, song_id, file_path):
        """Create folder for song if it doesn't exist and return path."""
        folder_name = self._get_song_folder_name(song_id, file_path)
        folder_path = os.path.join(self.base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def _load_metadata(self):
        """Load metadata from json file or create empty if doesn't exist."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save metadata to json file."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def save_segment(self, song_id, label, phrase_nr, start_time, end_time,
                     source_path, segmentation_sensitivity):
        """Save a new segment to the dataset with overwrite warning."""
        # Get/create song folder
        song_folder = self._ensure_song_folder(song_id, source_path)

        # Generate filename
        filename = f"{label}_ph{phrase_nr}.mp3"
        file_path = os.path.join(song_folder, filename)

        # Check if file exists
        if os.path.exists(file_path):
            if self.root:
                response = tk.messagebox.askyesno(
                    "File Exists",
                    f"File {filename} already exists in {os.path.basename(song_folder)}.\n"
                    "Do you want to overwrite it?"
                )
                if not response:
                    return None  # User chose not to overwrite
            else:
                print(f"Warning: File {filename} already exists. Skipping.")
                return None

        try:
            # Extract the segment
            segment = self.extract_segment(source_path, start_time, end_time)

            # Export segment to file
            segment.export(file_path, format="mp3")

            # Create metadata entry
            key = f"{song_id}_{label}_ph{phrase_nr}"
            self.metadata[key] = {
                "song_id": song_id,
                "label": label,
                "phrase_nr": phrase_nr,
                "segmentation_sensitivity": segmentation_sensitivity,
                "original_length": end_time - start_time,
                "timestamp": datetime.now().isoformat(),
                "folder_name": os.path.basename(song_folder)
            }

            # Save updated metadata
            self._save_metadata()

            return key

        except Exception as e:
            if self.root:
                tk.messagebox.showerror(
                    "Error",
                    f"Error saving segment: {str(e)}")
            else:
                print(f"Error saving segment: {str(e)}")
            return None

    def get_segment_metadata(self, song_id, label, phrase_nr):
        """Get metadata for a specific segment."""
        key = f"{song_id}_{label}_ph{phrase_nr}"
        return self.metadata.get(key)

    def list_segments(self):
        """List all segments in the dataset."""
        return list(self.metadata.keys())

    def get_segments_by_label(self, label):
        """Get all segments with a specific label."""
        return {k: v for k, v in self.metadata.items()
                if v["label"] == label}