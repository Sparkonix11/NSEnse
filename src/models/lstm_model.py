import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """LSTM model for sequence prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        Initialize LSTM model
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layer
            num_layers (int): Number of LSTM layers
            output_size (int): Size of output layer
            dropout (float): Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # We only need the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        # Apply sigmoid to get probabilities
        out = self.sigmoid(out)
        
        return out


class StockLSTMModel:
    """Stock price prediction using LSTM"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2, 
                 learning_rate=0.001, device=None):
        """
        Initialize Stock LSTM model
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layer
            num_layers (int): Number of LSTM layers
            output_size (int): Size of output layer
            dropout (float): Dropout rate
            learning_rate (float): Learning rate
            device (str): Device to use ('cuda' or 'cpu')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Create model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # For storing feature names
        self.feature_names = []
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=50, 
           feature_names=None, early_stopping=5, verbose=1):
        """
        Train the LSTM model
        
        Args:
            X_train (numpy.ndarray): Training features, shape (n_samples, seq_len, n_features)
            y_train (numpy.ndarray): Training targets
            X_val (numpy.ndarray, optional): Validation features
            y_val (numpy.ndarray, optional): Validation targets
            batch_size (int): Batch size
            epochs (int): Number of epochs
            feature_names (list, optional): Feature names
            early_stopping (int): Number of epochs with no improvement to wait before stopping
            verbose (int): Verbosity level
        
        Returns:
            self: The trained model instance
        """
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1)).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1)).to(self.device)
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Set model to training mode
            self.model.train()
            
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            # Batch training
            for batch_X, batch_y in train_loader:
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track loss
                train_loss += loss.item() * batch_X.size(0)
                
                # Track predictions
                train_preds.extend((outputs > 0.5).cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())
            
            # Average training loss
            train_loss /= len(X_train_tensor)
            
            # Calculate training accuracy
            train_acc = accuracy_score(train_targets, train_preds)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            if X_val is not None and y_val is not None:
                # Set model to evaluation mode
                self.model.eval()
                
                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(X_val_tensor)
                    val_loss = self.criterion(outputs, y_val_tensor).item()
                    
                    # Track predictions
                    val_preds = (outputs > 0.5).cpu().numpy()
                    val_targets = y_val_tensor.cpu().numpy()
                
                # Calculate validation accuracy
                val_acc = accuracy_score(val_targets, val_preds)
                
                # Track history
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Print progress
            if verbose and epoch % verbose == 0:
                if X_val is not None and y_val is not None:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                             f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make binary predictions
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).cpu().numpy()
        
        return predictions.flatten()
    
    def predict_proba(self, X):
        """
        Make probability predictions
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Probability predictions
        """
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = outputs.cpu().numpy()
        
        # Return probabilities for class 0 and 1
        return np.column_stack((1 - probabilities, probabilities)).flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test targets
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1] if len(y_test) > 0 else []
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Log results
        logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test precision: {metrics['precision']:.4f}")
        logger.info(f"Test recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1 score: {metrics['f1']:.4f}")
        
        return metrics
    
    def plot_history(self, figsize=(12, 5)):
        """
        Plot training history
        
        Args:
            figsize (tuple): Figure size
        """
        plt.figure(figsize=figsize)
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history and len(self.history['val_loss']) > 0:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        if 'val_acc' in self.history and len(self.history['val_acc']) > 0:
            plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save(self, filepath):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model state and metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'feature_names': self.feature_names,
            'model_config': {
                'input_size': self.model.lstm.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'output_size': self.model.fc.out_features,
                'dropout': self.model.lstm.dropout if self.model.num_layers > 1 else 0,
            }
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, device=None):
        """
        Load model from file
        
        Args:
            filepath (str): Path to load model from
            device (str): Device to load model to
            
        Returns:
            StockLSTMModel: Loaded model
        """
        # Load model state and metadata
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        # Create model instance with saved config
        model_instance = cls(
            input_size=checkpoint['model_config']['input_size'],
            hidden_size=checkpoint['model_config']['hidden_size'],
            num_layers=checkpoint['model_config']['num_layers'],
            output_size=checkpoint['model_config']['output_size'],
            dropout=checkpoint['model_config']['dropout'],
            device=device
        )
        
        # Load state dictionaries
        model_instance.model.load_state_dict(checkpoint['model_state_dict'])
        model_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history and feature names
        model_instance.history = checkpoint['history']
        model_instance.feature_names = checkpoint['feature_names']
        
        logger.info(f"Model loaded from {filepath}")
        
        return model_instance