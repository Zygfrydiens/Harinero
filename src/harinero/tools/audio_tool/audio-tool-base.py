import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import ruptures as rpt
from tanda_generator.preprocessor import Preprocessor
from tanda_generator.feature_extraction import (
    extract_energy, extract_rms, extract_spectral_contrast
)
from harinero_tanda_creator import HarineroTandaCreator
from AudioPlayer import AudioPlayer
import time
from tkinter import messagebox
from phrase_dataset import PhraseDataset


class SongProcessor:
    def __init__(self):
        # Configuration for the Preprocessor
        self.config = {
            'data': {
                'sampling_rate': 22050,
                'frame_time': 1,
                'selected_hop_time': 0.5,
                'dissimilar_hop_time': 0.1
            }
        }

        # Initialize the Preprocessor
        self.preprocessor = Preprocessor(self.config)

        # Initialize the database connection
        try:
            self.tanda_creator = HarineroTandaCreator('config.json')
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

    def get_song_by_id(self, song_id):
        """Get song information from database."""
        try:
            return self.tanda_creator.song_finder.get_song(song_id)
        except Exception as e:
            print(f"Error getting song {song_id}: {e}")
            raise

    def process_audio(self, file_path):
        """Process audio file and extract features."""
        try:
            # Load and slice the song
            frames, _ = self.preprocessor.slice_song(
                file_path,
                self.preprocessor.frame_size,
                self.preprocessor.hop_size_selected,
                label=1
            )

            # Extract features
            features = {
                'Energy': [],
                'RMS': [],
                'Spectral Contrast': [],
            }

            for frame in frames:
                features['Energy'].append(extract_energy(frame))
                features['RMS'].append(extract_rms(frame))
                features['Spectral Contrast'].append(
                    np.mean(extract_spectral_contrast(frame, self.config['data']['sampling_rate']))
                )

            # Normalize features
            normalized_features = {}
            for feature_name, feature_values in features.items():
                min_val = np.min(feature_values)
                max_val = np.max(feature_values)
                if max_val - min_val == 0:
                    normalized_features[feature_name] = [0] * len(feature_values)
                else:
                    normalized_features[feature_name] = (np.array(feature_values) - min_val) / (max_val - min_val)

            # Combine features
            feature_array = [
                normalized_features['RMS'],
                normalized_features['Spectral Contrast'],
                normalized_features['Energy']
            ]
            combined_feature = np.sum(feature_array, axis=0)

            return combined_feature

        except Exception as e:
            print(f"Error processing audio file {file_path}: {e}")
            raise

    def get_segments(self, combined_feature, pen_value=1.7):
        """Calculate segments using ruptures."""
        try:
            model = "l2"
            algo = rpt.Pelt(model=model).fit(combined_feature)
            change_points = algo.predict(pen=pen_value)
            return change_points
        except Exception as e:
            print(f"Error calculating segments: {e}")
            raise

    def get_random_song_id(self):
        """Get a random song ID from the database."""
        try:
            # Implement this based on your database API
            # For now, returning a dummy implementation
            import random
            return random.randint(1, 8000)  # Adjust range based on your database
        except Exception as e:
            print(f"Error getting random song: {e}")
            raise


class AudioToolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Segmentation Tool")

        # Initialize was_playing to False
        self.was_playing = False

        # Initialize song processor
        self.song_processor = SongProcessor()
        self.audio_player = AudioPlayer()

        # Initialize dataset handler
        self.dataset = PhraseDataset("./phrase_data")
        self.dataset.set_root(self.root)

        # Initialize state variables
        self.current_feature = None
        self.current_segments = None
        self.current_song = None

        # Configure main window
        self.root.geometry("1000x800")
        self.root.minsize(800, 600)

        # Create all frames and widgets
        self.create_layout()

        # Load initial song
        self.load_song(1)

    def create_layout(self):
        """Create the entire layout with explicit frame creation."""
        # 1. Navigation Section
        nav_frame = ttk.Frame(self.root, relief="groove", borderwidth=1)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        self.create_navigation_controls(nav_frame)

        # 2. Controls Section (includes playback and sensitivity)
        control_frame = ttk.Frame(self.root, relief="groove", borderwidth=1)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create separate frames for playback and sensitivity controls
        self.playback_frame = ttk.Frame(control_frame)
        self.playback_frame.pack(fill=tk.X, padx=5, pady=2)
        self.create_playback_controls(self.playback_frame)

        self.param_frame = ttk.Frame(control_frame)
        self.param_frame.pack(fill=tk.X, padx=5, pady=2)
        self.create_parameter_controls(self.param_frame)

        # 3. Graph Section
        graph_frame = ttk.Frame(self.root, relief="groove", borderwidth=1)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.create_graph(graph_frame)

        # 4. Segment Selection Section
        selection_frame = ttk.Frame(self.root, relief="groove", borderwidth=1)
        selection_frame.pack(fill=tk.X, padx=5, pady=5)
        self.create_phrase_selection(selection_frame)

    def create_phrase_selection(self, parent):
        """Create phrase selection controls with scrollable checkboxes."""
        # Add title
        ttk.Label(
            parent,
            text="Segment Selection",
            font=('TkDefaultFont', 9, 'bold')
        ).pack(anchor='w', padx=5, pady=(5, 0))

        # Create scrollable container for checkboxes
        self.checkbox_container = ttk.Frame(parent)
        self.checkbox_container.pack(fill=tk.X, expand=True, padx=5, pady=2)

        # Create canvas for scrolling
        self.checkbox_canvas = tk.Canvas(
            self.checkbox_container,
            height=75,
            highlightthickness=0,  # Remove canvas highlight
            bd=0  # Remove border
        )
        self.checkbox_scroll = ttk.Scrollbar(
            self.checkbox_container,
            orient="horizontal",
            command=self.checkbox_canvas.xview
        )

        # Create frame for checkboxes inside canvas
        self.checkbox_frame = ttk.Frame(self.checkbox_canvas)

        # Configure scrolling
        self.checkbox_canvas.configure(xscrollcommand=self.checkbox_scroll.set)
        self.checkbox_frame_window = self.checkbox_canvas.create_window(
            (0, 0),
            window=self.checkbox_frame,
            anchor="nw"
        )

        # Pack scrolling components
        self.checkbox_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.checkbox_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Create controls frame
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Save button on the left
        self.save_button = ttk.Button(
            control_frame,
            text="Save Selected",
            command=self.save_selected_segments,
            state='disabled'
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        # Label controls next to save button
        label_frame = ttk.Frame(control_frame)
        label_frame.pack(side=tk.LEFT, padx=(20, 5))  # Added padding before label

        ttk.Label(
            label_frame,
            text="Label:"
        ).pack(side=tk.LEFT, padx=2)

        self.label_var = tk.StringVar(value="1")
        self.label_spinbox = ttk.Spinbox(
            label_frame,
            from_=1,
            to=99,
            width=3,
            textvariable=self.label_var
        )
        self.label_spinbox.pack(side=tk.LEFT, padx=2)

    def create_frames(self):
        """Create main frames with improved visual organization."""
        # Navigation section with frame
        self.nav_outer_frame = ttk.Frame(self.root, relief="groove", borderwidth=1)
        self.nav_outer_frame.pack(fill=tk.X, padx=5, pady=5)

        self.nav_frame = ttk.Frame(self.nav_outer_frame)
        self.nav_frame.pack(fill=tk.X, padx=5, pady=5)

        # Single control section containing both playback and sensitivity controls
        self.controls_frame = ttk.Frame(self.root, relief="groove", borderwidth=1)
        self.controls_frame.pack(fill=tk.X, padx=5, pady=5)

        # Playback and parameter controls are now directly in the controls frame
        self.playback_frame = ttk.Frame(self.controls_frame)
        self.playback_frame.pack(fill=tk.X, padx=5, pady=2)

        self.param_frame = ttk.Frame(self.controls_frame)
        self.param_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Graph frame
        self.graph_outer_frame = ttk.Frame(self.root, relief="groove", borderwidth=1)
        self.graph_outer_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.graph_frame = ttk.Frame(self.graph_outer_frame)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Segment selection frame
        self.phrase_outer_frame = ttk.Frame(self.root, relief="groove", borderwidth=1)
        self.phrase_outer_frame.pack(fill=tk.X, padx=5, pady=5)

        self.phrase_frame = ttk.Frame(self.phrase_outer_frame)
        self.phrase_frame.pack(fill=tk.X, padx=5, pady=5)

    def create_navigation_controls(self, parent):
        """Create navigation controls."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(control_frame, text="Song ID:").pack(side=tk.LEFT, padx=5)

        self.song_id_var = tk.StringVar(value="1")
        self.song_id_entry = ttk.Entry(
            control_frame,
            textvariable=self.song_id_var,
            width=10
        )
        self.song_id_entry.pack(side=tk.LEFT, padx=5)

        # Add Load button
        ttk.Button(
            control_frame,
            text="Load",
            command=self.load_current_id
        ).pack(side=tk.LEFT, padx=2)

        # Also bind Enter key to load
        self.song_id_entry.bind('<Return>', lambda e: self.load_current_id())

        # Rest of the navigation buttons
        ttk.Button(
            control_frame,
            text="Previous",
            command=self.previous_song
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            control_frame,
            text="Next",
            command=self.next_song
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            control_frame,
            text="Random",
            command=self.random_song
        ).pack(side=tk.LEFT, padx=2)

        self.song_info_var = tk.StringVar(value="No song loaded")
        ttk.Label(
            control_frame,
            textvariable=self.song_info_var
        ).pack(side=tk.LEFT, padx=20)

    def create_graph(self, parent):
        """Create matplotlib graph with fixed size."""
        # Create frame for the graph
        self.graph_frame = ttk.Frame(parent)
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create matplotlib figure with fixed size
        self.fig = Figure(figsize=(10, 4), dpi=100)

        # Clear any existing subplots
        self.fig.clear()

        # Create single subplot
        self.ax = self.fig.add_subplot(111)

        # Set background colors
        self.ax.set_facecolor('#f8f9fa')
        self.fig.patch.set_facecolor('#f8f9fa')

        # Create initial plot
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.ax.plot(x, y, color='#2c3e50', linewidth=1.5)
        self.ax.grid(True, linestyle='--', alpha=0.3)

        # Add marker line
        self.marker_line = self.ax.axvline(x=0, color='#27ae60', linestyle='-', linewidth=1.5)

        # Create canvas only if it doesn't exist
        if not hasattr(self, 'canvas'):
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Set size and layout
        self.fig.set_tight_layout(True)
        self.initial_figsize = self.fig.get_size_inches()

        # Draw initial state
        self.canvas.draw()

    def create_playback_controls(self, parent):
        """Create playback controls with proper position updating."""
        # Button container
        button_frame = ttk.Frame(parent)
        button_frame.pack(side=tk.LEFT, padx=(0, 5))

        self.play_button = ttk.Button(
            button_frame,
            text="Play",
            command=self.toggle_playback
        )
        self.play_button.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            button_frame,
            text="Stop",
            command=self.stop_playback
        ).pack(side=tk.LEFT, padx=2)

        # Time display
        self.time_var = tk.StringVar(value="00:00 / 00:00")
        ttk.Label(
            button_frame,
            textvariable=self.time_var
        ).pack(side=tk.LEFT, padx=10)

        # Seek slider
        self.seek_var = tk.DoubleVar(value=0)
        self.seek_slider = ttk.Scale(
            parent,
            from_=0,
            to=100,  # Will be updated when loading song
            orient=tk.HORIZONTAL,
            variable=self.seek_var,
            command=lambda x: None  # Disable default callback
        )
        self.seek_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Bind slider events
        self.seek_slider.bind("<ButtonPress-1>", self.start_seeking)
        self.seek_slider.bind("<ButtonRelease-1>", self.end_seeking)
        self.seek_slider.bind("<B1-Motion>", self.on_slider_move)

    def on_slider_move(self, event):
        """Update marker position while dragging the seek slider."""
        position = self.seek_var.get()
        self.seek_audio(str(position))

    def create_parameter_controls(self, parent):
        """Create sensitivity controls."""
        param_frame = ttk.Frame(parent)
        param_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Label(
            param_frame,
            text="Segmentation Sensitivity:"
        ).pack(side=tk.LEFT, padx=5)

        self.pen_var = tk.DoubleVar(value=1.7)
        self.pen_slider = ttk.Scale(
            param_frame,
            from_=0.1,
            to=5.0,
            orient=tk.HORIZONTAL,
            variable=self.pen_var,
            command=self.update_segmentation
        )
        self.pen_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def create_phrase_buttons(self):
        """Create phrase selection interface with squared boxes."""
        # Main frame with border
        self.phrase_outer_frame = ttk.Frame(self.root, relief="groove", borderwidth=1)
        self.phrase_outer_frame.pack(fill=tk.X, padx=5, pady=5)

        # Title for the section
        title_frame = ttk.Frame(self.phrase_outer_frame)
        title_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(
            title_frame,
            text="Segment Selection",
            font=('TkDefaultFont', 9, 'bold')
        ).pack(side=tk.LEFT)

        # Frame for checkboxes with scrollbar if needed
        self.checkbox_container = ttk.Frame(self.phrase_outer_frame)
        self.checkbox_container.pack(fill=tk.X, expand=True, padx=5, pady=2)

        # Create a canvas for scrolling if many checkboxes
        self.checkbox_canvas = tk.Canvas(
            self.checkbox_container,
            height=75,  # Adjusted height
            highlightthickness=0  # Remove the highlight border
        )

        self.checkbox_canvas.bind("<Button-1>", lambda e: self.checkbox_canvas.master.focus_set())
        self.checkbox_scroll = ttk.Scrollbar(
            self.checkbox_container,
            orient="horizontal",
            command=self.checkbox_canvas.xview
        )
        self.checkbox_frame = ttk.Frame(self.checkbox_canvas)

        # Configure scrolling
        self.checkbox_canvas.configure(xscrollcommand=self.checkbox_scroll.set)
        self.checkbox_frame_window = self.checkbox_canvas.create_window(
            (0, 0),
            window=self.checkbox_frame,
            anchor="nw"
        )

        # Pack the scrolling components
        self.checkbox_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)
        self.checkbox_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Controls frame with separator
        ttk.Separator(self.phrase_outer_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=2)

        # Control frame for label selector and save button
        self.control_frame = ttk.Frame(self.phrase_outer_frame)
        self.control_frame.pack(fill=tk.X, padx=5, pady=2)

        # Label selector
        label_frame = ttk.Frame(self.control_frame)
        label_frame.pack(side=tk.LEFT)

        ttk.Label(
            label_frame,
            text="Label:"
        ).pack(side=tk.LEFT, padx=2)

        self.label_var = tk.StringVar(value="1")
        self.label_spinbox = ttk.Spinbox(
            label_frame,
            from_=1,
            to=99,
            width=3,
            textvariable=self.label_var,
            command=lambda: self.label_var.set(self.label_spinbox.get())
        )
        self.label_spinbox.pack(side=tk.LEFT, padx=2)

        # Save button (initially disabled)
        self.save_button = ttk.Button(
            self.control_frame,
            text="Save Selected",
            command=self.save_selected_segments,
            state='disabled'
        )
        self.save_button.pack(side=tk.RIGHT, padx=5)

        # Initialize empty list for checkbox variables
        self.phrase_vars = []

    def update_phrase_buttons(self):
        """Update checkboxes based on current segmentation with squared style."""
        # Clear existing checkboxes
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()
        self.phrase_vars = []

        if self.current_segments is None:
            return

        # Create new checkboxes for each segment
        for i in range(len(self.current_segments) - 1):
            var = tk.BooleanVar(value=False)
            var.trace('w', self.update_save_button_state)
            self.phrase_vars.append(var)

            # Create individual frame for each checkbox with border
            segment_frame = ttk.Frame(self.checkbox_frame, relief="solid", borderwidth=1)
            segment_frame.pack(side=tk.LEFT, padx=2, pady=2)

            # Calculate segment length in seconds
            start_time = self.segments_to_time(self.current_segments[i])
            end_time = self.segments_to_time(self.current_segments[i + 1])
            length = end_time - start_time

            # Use grid layout for more control over spacing
            inner_frame = ttk.Frame(segment_frame)
            inner_frame.pack(padx=4, pady=4)  # Add padding inside the border

            # Number label
            ttk.Label(
                inner_frame,
                text=f"Ph{i + 1}",
                width=4  # Fixed width for number
            ).pack()

            # Checkbox (using standard style)
            cb = ttk.Checkbutton(
                inner_frame,
                variable=var,
                command=lambda idx=i: self.on_checkbox_click(idx)
            )
            cb.pack()

            # Duration label
            ttk.Label(
                inner_frame,
                text=f"{length:.1f}s",
                width=6  # Fixed width for duration
            ).pack()

        # Update the canvas scroll region
        self.checkbox_frame.update_idletasks()
        self.checkbox_canvas.configure(
            scrollregion=self.checkbox_canvas.bbox("all")
        )

        # Initial button state
        self.update_save_button_state()

    def update_save_button_state(self, *args):
        """Update save button state based on checkbox selection."""
        any_selected = any(var.get() for var in self.phrase_vars)

        if any_selected:
            self.save_button.configure(state='normal')
        else:
            self.save_button.configure(state='disabled')

    def setup_styles(self):
        """Setup custom styles for widgets."""
        style = ttk.Style()

        # Style for info labels
        style.configure(
            'Info.TLabel',
            padding=5,
            font=('TkDefaultFont', 9)
        )

        # Remove focus highlighting
        style.configure('TFrame', focuscolor=style.configure(".")["background"])
        style.configure('TButton', focuscolor=style.configure(".")["background"])
        style.configure('TCheckbutton', focuscolor=style.configure(".")["background"])

        # If you're using any other ttk widgets that show focus, add them here
        # For example:
        style.configure('TSpinbox', focuscolor=style.configure(".")["background"])
        """Setup custom styles for the matplotlib figure."""
        plt.style.use('seaborn-v0_8-whitegrid')  # Use a clean base style
        self.fig.set_dpi(100)  # Ensure crisp rendering
        # Also remove highlight thickness from Canvas
        self.checkbox_canvas.configure(highlightthickness=0)

    def load_song(self, song_id):
        """Load and process a song by ID with position update callback."""
        try:
            # Get song from database
            self.current_song = self.song_processor.get_song_by_id(song_id)

            # Update song info display
            self.song_info_var.set(f"{self.current_song.name} - {self.current_song.singer}")

            # Process audio features
            self.current_feature = self.song_processor.process_audio(self.current_song.file_path)

            # Load audio file and set up position callback
            if self.audio_player.load(self.current_song.file_path):
                # Update seek slider range
                self.seek_slider.configure(to=self.audio_player.duration)
                self.seek_var.set(0)
                self.update_time_display(0)

                # Set up position update callback
                self.audio_player.set_update_callback(self.update_playback_position)

            # Get initial segmentation
            self.current_segments = self.song_processor.get_segments(
                self.current_feature,
                self.pen_var.get()
            )

            # Update display
            self.update_graph()
            self.update_phrase_buttons()

            # Reset playback controls
            self.play_button.configure(text="Play")
            self.audio_player.stop()
            self.update_playback_position(0)

        except Exception as e:
            print(f"Error loading song {song_id}: {e}")
            self.song_info_var.set(f"Error loading song {song_id}")

    def load_current_id(self):
        """Load song based on current ID in entry box."""
        try:
            song_id = int(self.song_id_var.get())
            self.load_song(song_id)
        except ValueError:
            tk.messagebox.showerror("Error", "Please enter a valid song ID number")

    def toggle_playback(self):
        """Toggle between play and pause."""
        if not self.current_song:
            return

        if self.audio_player.is_playing:
            self.audio_player.pause()
            self.play_button.configure(text="Play")
        else:
            self.audio_player.play()
            self.play_button.configure(text="Pause")

    def stop_playback(self):
        """Stop playback and reset position."""
        self.audio_player.stop()
        self.play_button.configure(text="Play")
        self.update_playback_position(0)

    def seek_audio(self, value):
        """Handle seek slider movement."""
        try:
            position = float(value)

            # Update the audio position
            self.audio_player.position = position  # Update stored position

            # Update marker position
            if self.current_feature is not None and hasattr(self, 'marker_line'):
                frame_pos = int((position / self.audio_player.duration) * len(self.current_feature))
                frame_pos = max(0, min(frame_pos, len(self.current_feature) - 1))

                self.marker_line.set_xdata([frame_pos, frame_pos])
                self.canvas.draw_idle()  # Use draw_idle for better performance

            # Update time display
            current = time.strftime('%M:%S', time.gmtime(position))
            total = time.strftime('%M:%S', time.gmtime(self.audio_player.duration))
            self.time_var.set(f"{current} / {total}")

            # If audio is playing, seek to the new position
            if self.audio_player.is_playing:
                self.audio_player.seek(position)

        except Exception as e:
            print(f"Error seeking audio: {e}")

    def start_seeking(self, event):
        """Handle start of seek slider interaction."""
        if self.audio_player.is_playing:
            self.audio_player.pause()
            self.was_playing = True
        else:
            self.was_playing = False

    def end_seeking(self, event):
        """Handle end of seek slider interaction."""
        position = self.seek_var.get()
        # Update the audio position and marker line
        self.seek_audio(str(position))

        # Ensure was_playing is defined
        if hasattr(self, 'was_playing') and self.was_playing:
            self.audio_player.play()

    def update_playback_position(self, position):
        """Update GUI elements based on playback position."""
        try:
            # Update seek slider without triggering events
            self.seek_var.set(position)

            # Update time display
            self.update_time_display(position)

            # Update marker position
            if self.current_feature is not None and hasattr(self, 'marker_line'):
                frame_pos = int((position / self.audio_player.duration) * len(self.current_feature))
                frame_pos = max(0, min(frame_pos, len(self.current_feature) - 1))

                self.marker_line.set_xdata([frame_pos, frame_pos])
                self.canvas.draw_idle()

        except Exception as e:
            print(f"Error updating playback position: {e}")

    def update_time_display(self, position):
        """Update the time display label."""
        current = time.strftime('%M:%S', time.gmtime(position))
        total = time.strftime('%M:%S', time.gmtime(self.audio_player.duration))
        self.time_var.set(f"{current} / {total}")

    def update_marker_position(self, position):
        """Update the graph marker position."""
        if hasattr(self, 'marker_line'):
            if self.current_feature is not None:
                x_pos = (position / self.audio_player.duration) * len(self.current_feature)
                self.marker_line.set_xdata([x_pos, x_pos])
                self.canvas.draw_idle()

    def update_graph(self, current_position=None):
        """Update graph with current marker position."""
        if self.current_feature is None:
            return

        # Store current figure size
        current_size = self.fig.get_size_inches()

        # Clear the figure completely
        self.fig.clear()

        # Create new subplot
        self.ax = self.fig.add_subplot(111)

        # Restore background and grid
        self.ax.set_facecolor('#f8f9fa')
        self.fig.patch.set_facecolor('#f8f9fa')
        self.ax.grid(True, linestyle='--', alpha=0.3)

        # Plot the feature
        self.ax.plot(self.current_feature, color='#2c3e50',
                     label='Combined Feature', linewidth=1.5)

        if self.current_segments is not None:
            for i in range(len(self.current_segments) - 1):
                start = self.current_segments[i]
                end = self.current_segments[i + 1]
                mean_value = np.mean(self.current_feature[start:end])

                is_selected = False
                if hasattr(self, 'phrase_vars') and i < len(self.phrase_vars):
                    is_selected = self.phrase_vars[i].get()

                line_color = '#3498db' if is_selected else '#e74c3c'
                alpha = 0.15 if is_selected else 0.0

                # Draw mean line
                self.ax.hlines(mean_value, start, end, colors=line_color,
                               linewidth=2, alpha=0.8)

                # Draw vertical boundaries
                self.ax.vlines(start, min(self.current_feature), max(self.current_feature),
                               colors=line_color, linewidth=1.5, alpha=0.5,
                               linestyles='--')

                if is_selected:
                    self.ax.fill_between([start, end],
                                         min(self.current_feature),
                                         max(self.current_feature),
                                         color=line_color, alpha=alpha)

                # Add labels with background
                label_y = max(self.current_feature) + 0.02 * (max(self.current_feature) - min(self.current_feature))
                label_x = (start + end) / 2

                bbox_props = dict(boxstyle="round,pad=0.3",
                                  fc='white',
                                  ec=line_color,
                                  alpha=0.8)

                self.ax.text(label_x, label_y, f'ph{i + 1}',
                             rotation=90,
                             horizontalalignment='center',
                             verticalalignment='bottom',
                             fontsize=9,
                             bbox=bbox_props)

        # Add marker line at current position
        if current_position is None:
            current_position = float(self.seek_var.get())

        # Calculate frame position for marker
        if self.audio_player.duration > 0:  # Prevent division by zero
            frame_pos = int((current_position / self.audio_player.duration) * len(self.current_feature))
            frame_pos = max(0, min(frame_pos, len(self.current_feature) - 1))
        else:
            frame_pos = 0

        self.marker_line = self.ax.axvline(x=frame_pos,
                                           color='#27ae60',
                                           linestyle='-',
                                           linewidth=1.5)

        # Update graph appearance
        self.ax.set_title(f"Audio Features - {self.current_song.name}",
                          pad=20, fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Frame Index", labelpad=10)
        self.ax.set_ylabel("Feature Value", labelpad=10)

        # Maintain margins
        self.ax.margins(y=0.15)

        # Restore original size
        self.fig.set_size_inches(current_size)
        self.fig.set_tight_layout(True)

        # Update the canvas
        self.canvas.draw()

    def previous_song(self):
        """Load the previous song."""
        current = int(self.song_id_var.get())
        if current > 1:
            new_id = current - 1
            self.song_id_var.set(str(new_id))
            self.load_song(new_id)

    def next_song(self):
        """Load the next song."""
        current = int(self.song_id_var.get())
        new_id = current + 1
        self.song_id_var.set(str(new_id))
        self.load_song(new_id)

    def random_song(self):
        """Load a random song."""
        new_id = self.song_processor.get_random_song_id()
        self.song_id_var.set(str(new_id))
        self.load_song(new_id)

    def save_segment(self, segment_index):
        """Save the selected segment."""
        if self.current_segments is None or self.current_song is None:
            return

        start = self.current_segments[segment_index]
        end = self.current_segments[segment_index + 1]
        print(f"Saving segment {segment_index + 1}: {start} to {end}")
        # TODO: Implement actual saving logic

    def segments_to_time(self, segment_index):
        """Convert segment index to time in seconds."""
        if self.current_feature is not None and self.audio_player.duration > 0:
            return (segment_index / len(self.current_feature)) * self.audio_player.duration
        return 0

    def on_checkbox_click(self, index):
        """Handle checkbox clicks - update graph to show selection."""
        if self.phrase_vars[index].get():
            print(f"Selected segment {index + 1}")
        self.update_graph()  # Redraw graph with new selection state

    def save_selected_segments(self):
        """Save all selected segments with the current label."""
        if not self.current_segments or not self.current_song:
            return

        label = self.label_var.get()
        saved_segments = []
        skipped_segments = []

        try:
            for i, var in enumerate(self.phrase_vars):
                if var.get():  # If checkbox is checked
                    # Get segment boundaries
                    start = self.current_segments[i]
                    end = self.current_segments[i + 1]

                    # Convert to actual time
                    start_time = self.segments_to_time(start)
                    end_time = self.segments_to_time(end)

                    # Save the segment
                    key = self.dataset.save_segment(
                        song_id=self.current_song.song_id,
                        label=int(label),
                        phrase_nr=i + 1,
                        start_time=start_time,
                        end_time=end_time,
                        source_path=self.current_song.file_path,
                        segmentation_sensitivity=float(self.pen_var.get())
                    )

                    if key:
                        saved_segments.append(key)
                    else:
                        skipped_segments.append(i + 1)

            # Show summary message
            if saved_segments or skipped_segments:
                message = []
                if saved_segments:
                    message.append(f"Saved {len(saved_segments)} segments with label {label}")
                if skipped_segments:
                    message.append(f"Skipped phrases: {', '.join(map(str, skipped_segments))}")

                tk.messagebox.showinfo("Save Complete", "\n".join(message))

                # Clear checkboxes after successful save
                for var in self.phrase_vars:
                    var.set(False)

        except Exception as e:
            tk.messagebox.showerror("Error", f"Error saving segments: {str(e)}")

    def update_segmentation(self, value):
        """Update segmentation when pen parameter changes."""
        if self.current_feature is None:
            return

        # Recalculate segments with new pen value
        self.current_segments = self.song_processor.get_segments(
            self.current_feature,
            float(value)
        )

        # Update display
        self.update_graph()
        self.update_phrase_buttons()


def main():
    root = tk.Tk()
    app = AudioToolGUI(root)
    root.mainloop()


main()
