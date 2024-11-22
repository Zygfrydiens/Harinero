from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import tensorflow as tf


@dataclass
class Triplet:
    """Represents a training triplet.

    Attributes:
        anchor: Anchor sample
        positive: Similar to anchor
        negative: Dissimilar to anchor
    """
    anchor: np.ndarray
    positive: np.ndarray
    negative: np.ndarray


class TripletGenerator:
    """Generates triplets for training Siamese networks."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """Initialize generator with features and labels.

        Args:
            features: Array of shape (n_samples, timesteps, channels)
            labels: Binary labels array (n_samples,)
        """
        self.features = features
        self.labels = labels
        self._similar_idx = np.where(labels == 1)[0]
        self._dissimilar_idx = np.where(labels == 0)[0]

    def generate_triplets(self,
                          n_triplets: Optional[int] = None,
                          batch_size: int = 32
                          ) -> tf.data.Dataset:
        """Create tf.data.Dataset of triplets.

        Args:
            n_triplets: Number of triplets to generate
            batch_size: Batch size for training

        Returns:
            TensorFlow dataset of triplets
        """
        if n_triplets is None:
            n_triplets = min(len(self._similar_idx),
                             len(self._dissimilar_idx))

        triplets = self._create_triplets(n_triplets)

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            triplets.anchor,
            triplets.positive,
            triplets.negative
        ))

        return dataset.batch(batch_size)

    def _create_triplets(self, n_triplets: int) -> Triplet:
        """Create specified number of triplets.

        Args:
            n_triplets: Number of triplets to generate

        Returns:
            Triplet object containing arrays
        """
        shape = (n_triplets,) + self.features.shape[1:]

        anchor = np.zeros(shape)
        positive = np.zeros(shape)
        negative = np.zeros(shape)

        for i in range(n_triplets):
            # Get anchor and positive from similar pairs
            a_idx, p_idx = np.random.choice(
                self._similar_idx, 2, replace=False
            )
            # Get negative from dissimilar pairs
            n_idx = np.random.choice(self._dissimilar_idx)

            anchor[i] = self.features[a_idx]
            positive[i] = self.features[p_idx]
            negative[i] = self.features[n_idx]

        return Triplet(anchor, positive, negative)