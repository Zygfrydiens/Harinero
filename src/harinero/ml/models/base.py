"""Base model interface for Harinero Tanda models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import numpy.typing as npt


class BaseModel(ABC):
    """Abstract base class for all song models.

    Reason:
        Ensure consistent interface across different model implementations
        (Keras, PyTorch) and enforce required methods.
    """

    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config

    @abstractmethod
    def train_on_data(self,
                      X: npt.NDArray[np.float32],
                      y: npt.NDArray[np.float32],
                      X_val: Optional[npt.NDArray[np.float32]] = None,
                      y_val: Optional[npt.NDArray[np.float32]] = None
                      ) -> Tuple[Any, Any]:
        """Train the model on provided data.

        Args:
            X: Training features
            y: Training labels
            X_val: Optional validation features
            y_val: Optional validation labels

        Returns:
            Tuple containing training history and evaluation results
        """
        pass

    @abstractmethod
    def predict_on_data(self, X: npt.NDArray[np.float32]) -> float:
        """Make predictions on provided data.

        Args:
            X: Input features to predict on

        Returns:
            Prediction score between 0 and 1
        """
        pass

    @abstractmethod
    def save_model(self, song_id: int, directory: str = 'models') -> None:
        """Save model to disk.

        Args:
            song_id: ID of the song model is trained on
            directory: Directory to save the model
        """
        pass