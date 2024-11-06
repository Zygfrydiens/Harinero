import pygame
import time
from threading import Thread
import numpy as np


class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.is_playing = False
        self.current_file = None
        self.position = 0  # Position in seconds
        self.duration = 0  # Duration in seconds
        self.update_callback = None
        self._update_thread = None
        self.start_time = None  # Initialize start_time

    def load(self, file_path):
        """Load a new audio file."""
        try:
            if pygame.mixer.music.get_busy():
                self.stop()

            pygame.mixer.music.load(file_path)
            self.current_file = file_path

            # Get audio duration
            temp_sound = pygame.mixer.Sound(file_path)
            self.duration = temp_sound.get_length()
            del temp_sound

            self.position = 0
            return True

        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False

    def play(self):
        """Start or resume playback."""
        if not self.current_file:
            return

        if not self.is_playing:
            # Start playback from the current position
            pygame.mixer.music.play(start=self.position)
            self.is_playing = True

            # Adjust start_time
            self.start_time = time.time() - self.position
            self._start_update_thread()

    def pause(self):
        """Pause playback."""
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
            # Update position based on elapsed time
            self.position = time.time() - self.start_time

    def stop(self):
        """Stop playback."""
        pygame.mixer.music.stop()
        self.is_playing = False
        self.position = 0

    def seek(self, position):
        """Seek to a specific position in seconds."""
        if not self.current_file:
            return

        print(f"Seeking to position: {position}")  # Debug print
        was_playing = self.is_playing
        pygame.mixer.music.stop()
        self.position = position

        if was_playing:
            print(f"Resuming playback at: {position}")  # Debug print
            pygame.mixer.music.play(start=self.position)
            self.is_playing = True

            # Adjust start_time
            self.start_time = time.time() - self.position
            self._start_update_thread()
        else:
            self.is_playing = False

    def set_update_callback(self, callback):
        """Set callback function for position updates."""
        self.update_callback = callback

    def _start_update_thread(self):
        """Start the thread that updates the current position."""
        if self._update_thread is not None and self._update_thread.is_alive():
            return

        self._update_thread = Thread(target=self._update_position, daemon=True)
        self._update_thread.start()

    def _update_position(self):
        """Update the current position while playing."""
        while self.is_playing:
            if not pygame.mixer.music.get_busy():
                self.is_playing = False
                self.position = 0
                if self.update_callback:
                    self.update_callback(0)
                break

            current_pos = time.time() - self.start_time
            if current_pos >= self.duration:
                self.stop()
                break

            self.position = current_pos
            if self.update_callback:
                self.update_callback(current_pos)

            time.sleep(0.03)  # Update more frequently (about 30fps)