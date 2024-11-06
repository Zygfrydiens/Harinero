import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class SongModel(nn.Module):
    def __init__(self, config):
        super(SongModel, self).__init__()
        self.config = config
        sampling_rate = config['data']['sampling_rate']
        frame_time = config['data']['frame_time']
        self.input_shape = sampling_rate * frame_time
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(in_features=64 * (self.input_shape - 4), out_features=64),  # Adjust flattened size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid()
        )
        return model

    def forward(self, x):
        return self.model(x)

    def train_on_data(self, X_train, y_train, X_val=None, y_val=None, epochs=20, batch_size=32, learning_rate=0.001):
        # Convert inputs to torch tensors and move to the device
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        if X_val is not None and y_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        # Create DataLoader for training and validation
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Define loss function and optimizer
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Lists to store losses and accuracies for plotting
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        # Training loop
        self.train()  # Set the model to training mode
        for epoch in range(epochs):
            train_loss = 0.0
            correct_train = 0
            total_train = 0

            for inputs, labels in train_loader:
                inputs = inputs.unsqueeze(1).to(self.device)  # Add channel dimension and move to device
                labels = labels.to(self.device)  # Move labels to device

                optimizer.zero_grad()  # Clear gradients
                outputs = self.model(inputs)  # Forward pass
                loss = criterion(outputs.squeeze(), labels)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                train_loss += loss.item()

                # Calculate accuracy
                predicted = (outputs.squeeze() > 0.5).float()  # Convert to binary predictions
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

            # Average loss and accuracy for the epoch
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = correct_train / total_train
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

            # Validation
            if X_val is not None and y_val is not None:
                self.eval()  # Set the model to evaluation mode
                val_loss = 0.0
                correct_val = 0
                total_val = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.unsqueeze(1).to(self.device)  # Add channel dimension and move to device
                        labels = labels.to(self.device)  # Move labels to device
                        outputs = self.model(inputs)  # Forward pass
                        loss = criterion(outputs.squeeze(), labels)  # Compute loss
                        val_loss += loss.item()

                        # Calculate accuracy
                        predicted = (outputs.squeeze() > 0.5).float()  # Convert to binary predictions
                        correct_val += (predicted == labels).sum().item()
                        total_val += labels.size(0)

                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = correct_val / total_val
                val_losses.append(avg_val_loss)
                val_accuracies.append(val_accuracy)

                # Print statistics
                print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            else:
                # Print statistics without validation
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

        # Plotting learning curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        if X_val is not None:
            plt.plot(val_losses, label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        if X_val is not None:
            plt.plot(val_accuracies, label='Val Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        # Plotting learning curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        if X_val is not None:
            plt.plot(val_losses, label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        if X_val is not None:
            plt.plot(val_accuracies, label='Val Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def predict_on_data(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, self.input_shape).to(self.device)
        with torch.no_grad():
            outputs = self(X).squeeze()
        return torch.mean((outputs > 0.5).float()).item()

    def _plot_learning_curves(self, history):
        df = pd.DataFrame(history)
        df.plot(figsize=(10, 6))
        plt.title('Training and Validation Metrics')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.grid(True)
        plt.show()

    def save_model(self, song_id, directory='models'):
        # Convert model parameters to half precision
        model_half = self.model.half()

        # Ensure the directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Define the file path
        filename = f"{song_id}_model.pth"
        file_path = os.path.join(directory, filename)

        # Save the model's state_dict with half precision
        torch.save(model_half.state_dict(), file_path)
        print(f"Model saved to {file_path} in half precision format")

    def load_model(self, file_path):
        try:
            state_dict = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.float()
            print(f"Model loaded successfully from {file_path} and converted to full precision.")
        except Exception as e:
            print(f"Failed to load the model from {file_path}: {e}")