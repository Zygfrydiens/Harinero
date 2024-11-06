from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import numpy.typing as npt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

from .base import BaseModel


class KerasSongModel(BaseModel):
    """Keras implementation of song similarity model.

    Reason:
        Provides a CNN-based model for learning song similarities using Keras.
        Uses binary classification approach with similarity scores.

    Attributes:
        config: Model configuration dictionary
        input_shape: Shape of input audio segments
        model: Keras Sequential model instance
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Keras song model.

        Args:
            config: Configuration dictionary containing:
                - data.sampling_rate: Audio sampling rate
                - data.frame_time: Frame duration in seconds
                - data.test_size: Validation split ratio
                - data.random_state: Random seed
                - model.epochs: Number of training epochs
                - model.patience: Early stopping patience
                - model.baseline_accuracy: Minimum accuracy threshold

        Raises:
            ValueError: If required config parameters are missing
        """
        super().__init__(config)
        self._validate_config()

        sampling_rate = self.config['data']['sampling_rate']
        frame_time = self.config['data']['frame_time']
        self.input_shape = (sampling_rate * frame_time, 1)
        self.model = self._build_model()

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        required_params = {
            'data': ['sampling_rate', 'frame_time', 'test_size', 'random_state'],
            'model': ['epochs', 'patience', 'baseline_accuracy']
        }

        for section, params in required_params.items():
            if section not in self.config:
                raise ValueError(f"Missing config section: {section}")
            for param in params:
                if param not in self.config[section]:
                    raise ValueError(f"Missing config parameter: {section}.{param}")

    def _build_model(self):
        """Build and compile the Keras model.

        Architecture designed to handle large input sequences with multiple
        dimensionality reduction steps.
        """
        model = Sequential()

        # First conv block with aggressive pooling
        model.add(Conv1D(
            filters=32,
            kernel_size=3,
            strides=1,
            activation='relu',
            input_shape=self.input_shape
        ))
        model.add(keras.layers.MaxPooling1D(pool_size=4))
        model.add(Dropout(0.5))

        # Second conv block with pooling
        model.add(Conv1D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation='relu'
        ))
        model.add(keras.layers.MaxPooling1D(pool_size=4))
        model.add(Dropout(0.5))

        # Global pooling to handle variable length inputs
        model.add(keras.layers.GlobalAveragePooling1D())

        # Dense layers
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=1, activation='sigmoid'))

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_on_data(
            self,
            X: npt.NDArray[np.float32],
            y: npt.NDArray[np.float32],
            X_val: Optional[npt.NDArray[np.float32]] = None,
            y_val: Optional[npt.NDArray[np.float32]] = None
    ) -> Tuple[keras.callbacks.History, List[float]]:
        """Train the model on provided data.

        Args:
            X: Training features
            y: Training labels
            X_val: Optional validation features (if None, will split from X)
            y_val: Optional validation labels (if None, will split from y)

        Returns:
            Tuple containing:
                - Training history
                - Evaluation results [loss, accuracy]
        """
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = self._prepare_train_val_split(X, y)
        else:
            X_train, y_train = X, y

        X_train = self._reshape_input(X_train)
        X_val = self._reshape_input(X_val)

        callbacks = self._create_callbacks()

        history = self.model.fit(
            X_train, y_train,
            epochs=self.config['model']['epochs'],
            callbacks=callbacks,
            validation_data=(X_val, y_val),
            verbose=1
        )

        evaluation_results = self.evaluate(X_val, y_val)
        self._log_training_results(history, evaluation_results)
        self._plot_learning_curves(history)

        return history, evaluation_results

    def _log_training_results(self, history: keras.callbacks.History, evaluation_results: List[float]) -> None:
        """Log training results for monitoring.

        Args:
            history: Training history
            evaluation_results: Evaluation metrics [loss, accuracy]
        """
        print(f"\nTraining completed:")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"\nTest evaluation:")
        print(f"Test loss: {evaluation_results[0]:.4f}")
        print(f"Test accuracy: {evaluation_results[1]:.4f}")

    def _create_callbacks(self) -> List[keras.callbacks.Callback]:
        """Create training callbacks.

        Returns:
            List of Keras callbacks for training
        """
        return [
            keras.callbacks.EarlyStopping(
                monitor='accuracy',
                mode='max',
                patience=self.config['model']['patience'],
                baseline=self.config['model']['baseline_accuracy'],
                verbose=1
            )
        ]

    def _prepare_train_val_split(
            self,
            X: npt.NDArray[np.float32],
            y: npt.NDArray[np.float32]
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32],
               npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Prepare train-validation split.

        Args:
            X: Input features
            y: Input labels

        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        return train_test_split(
            X, y,
            test_size=self.config['data']['test_size'],
            stratify=y,
            random_state=self.config['data']['random_state']
        )

    def _reshape_input(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Reshape input data for the model.

        Args:
            X: Input data

        Returns:
            Reshaped input data
        """
        return X.reshape(X.shape[0], X.shape[1], 1)

    def evaluate(
            self,
            X_test: npt.NDArray[np.float32],
            y_test: npt.NDArray[np.float32]
    ) -> List[float]:
        """Evaluate the model on test data.

        Args:
            X_test: Test features of shape (n_samples, timesteps)
            y_test: Test labels of shape (n_samples,)

        Returns:
            List containing [loss, accuracy] on test data
        """
        X_test = self._reshape_input(X_test)  # Reuse helper method for reshaping
        return self.model.evaluate(X_test, y_test, verbose=1)

    def predict_on_data(self, X: npt.NDArray[np.float32]) -> float:
        """Make predictions on input data.

        Args:
            X: Input features of shape (n_samples, timesteps)

        Returns:
            Float between 0 and 1 representing similarity score
            (mean of rounded predictions)
        """
        X = self._reshape_input(X)
        predictions = self.model.predict(X, verbose=0)  # Added verbose=0 to reduce output noise
        return float(np.mean(np.round(predictions, 3)))  # Explicit float conversion

    def _plot_learning_curves(
            self,
            history: keras.callbacks.History,
            save_path: Optional[Path] = None
    ) -> None:
        """Plot training and validation metrics over epochs.

        Args:
            history: Keras training history object
            save_path: Optional path to save the plot. If None, displays plot

        Note:
            Displays accuracy and loss curves for both training and validation
        """
        metrics_df = pd.DataFrame(history.history)

        plt.figure(figsize=(12, 5))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(metrics_df['loss'], label='Training Loss')
        if 'val_loss' in metrics_df.columns:
            plt.plot(metrics_df['val_loss'], label='Validation Loss')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(metrics_df['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in metrics_df.columns:
            plt.plot(metrics_df['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_model(self, song_id: int, directory: str = 'models') -> None:
        """Save model to disk.

        Args:
            song_id: ID of the song model is trained on
            directory: Directory to save the model

        Raises:
            OSError: If directory creation or model saving fails
        """
        save_path = Path(directory)
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            model_path = save_path / f"{song_id}_model.h5"
            self.model.save(model_path)
        except Exception as e:
            raise OSError(f"Failed to save model: {e}")
