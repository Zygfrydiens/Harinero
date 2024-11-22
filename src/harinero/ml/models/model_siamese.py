from typing import Tuple, Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from .base import BaseModel


class SiameseSongModel():
    """Siamese network for audio similarity learning using triplet loss."""

    def __init__(self,
                 input_shape: Tuple[int, int],
                 embedding_dim: int = 128,
                 margin: float = 0.5):
        """Initialize network.

        Args:
            input_shape: Input shape (timesteps, channels)
            embedding_dim: Embedding space dimension
            margin: Triplet loss margin
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.encoder = self._build_encoder()
        self.model = self._build_model()

    def _build_encoder(self) -> keras.Model:
        """Build encoder network optimized for MFCC features.

        Architecture designed to:
        1. Capture temporal patterns in each coefficient
        2. Learn relationships between coefficients
        3. Extract hierarchical features effectively
        """
        inputs = layers.Input(shape=self.input_shape)

        # Batch normalization on input
        x = layers.BatchNormalization()(inputs)

        # First conv block - capture local patterns
        x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)

        # Second conv block - medium-scale patterns
        x = layers.Conv1D(128, kernel_size=5, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)

        # Third conv block - larger patterns
        x = layers.Conv1D(256, kernel_size=7, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Global pooling to handle variable length
        x = layers.GlobalAveragePooling1D()(x)

        # Dense layers for final embedding
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(self.embedding_dim)(x)

        # L2 normalization
        outputs = layers.Lambda(
            lambda x: tf.math.l2_normalize(x, axis=1),
            name='l2_norm'
        )(x)

        return keras.Model(inputs, outputs, name="encoder")

    def _build_model(self) -> keras.Model:
        """Build the complete triplet model."""
        anchor_input = layers.Input(shape=self.input_shape, name="anchor_input")
        positive_input = layers.Input(shape=self.input_shape, name="positive_input")
        negative_input = layers.Input(shape=self.input_shape, name="negative_input")

        # Shared encoder for all inputs
        anchor_encoding = self.encoder(anchor_input)
        positive_encoding = self.encoder(positive_input)
        negative_encoding = self.encoder(negative_input)

        # Merge for loss calculation
        merged = layers.Concatenate(axis=1)([
            anchor_encoding,
            positive_encoding,
            negative_encoding
        ])

        return keras.Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=merged,
            name="triplet_siamese"
        )

    def _triplet_loss(self, _, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate triplet loss."""
        # Split embeddings
        anchor = y_pred[:, :self.embedding_dim]
        positive = y_pred[:, self.embedding_dim:self.embedding_dim * 2]
        negative = y_pred[:, self.embedding_dim * 2:]

        # Calculate distances and loss
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + self.margin, 0.0))

    def compile(self, learning_rate: float = 0.0001):
        """Compile model with triplet loss."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=self._triplet_loss
        )

    def train(self,
              dataset: tf.data.Dataset,
              validation_data: Optional[tf.data.Dataset] = None,
              epochs: int = 20,
              **kwargs) -> keras.callbacks.History:
        """Train model using tf.data.Dataset of triplets."""
        return self.model.fit(
            dataset,
            epochs=epochs,
            validation_data=validation_data,
            **kwargs
        )

    def get_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Get embedding for audio sample."""
        if audio.ndim == 2:
            audio = np.expand_dims(audio, 0)
        return self.encoder.predict(audio, verbose=0)

    def predict_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Calculate cosine similarity between two audio samples."""
        emb1 = self.get_embedding(audio1)
        emb2 = self.get_embedding(audio2)

        similarity = np.sum(emb1 * emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        return float(similarity)

    def save_weights(self, encoder_path: str, model_path: str):
        """Save both encoder and model weights.

        Args:
            encoder_path: Path to save encoder weights
            model_path: Path to save full model weights

        Raises:
            ValueError: If saving fails
        """
        try:
            self.encoder.save_weights(str(encoder_path))
            self.model.save_weights(str(model_path))
        except Exception as e:
            raise ValueError(f"Failed to save weights: {str(e)}")

    def load_weights(self, encoder_path: str, model_path: str):
        """Load both encoder and model weights.

        Args:
            encoder_path: Path to encoder weights
            model_path: Path to full model weights

        Raises:
            ValueError: If loading fails
        """
        try:
            self.encoder.load_weights(str(encoder_path))
            self.model.load_weights(str(model_path))
        except Exception as e:
            raise ValueError(f"Failed to load weights: {str(e)}")