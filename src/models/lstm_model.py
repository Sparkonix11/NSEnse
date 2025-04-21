import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import shap

logger = logging.getLogger(__name__)

class LSTMNetwork(nn.Module):
    """LSTM Neural Network architecture for stock prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, bidirectional=True):
        """
        Initialize LSTM model architecture
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layer
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate for regularization
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.directions, 1),
            nn.Softmax(dim=1)
        )
        
        # Fully connected layers with batch normalization and dropout
        self.fc1 = nn.Linear(hidden_size * self.directions, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout * 0.5)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for faster convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input data with shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output probabilities
        """
        # Initial hidden states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.directions, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))
        
        # Attention mechanism
        attention_weights = self.attention(out)
        context_vector = torch.sum(attention_weights * out, dim=1)
        
        # Fully connected layers
        out = self.fc1(context_vector)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.sigmoid(out)
        
        return out


class StockLSTMModel:
    """Stock price movement prediction using LSTM model"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, learning_rate=0.001, 
                 bidirectional=True, device=None):
        """
        Initialize StockLSTMModel
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden layer
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            learning_rate (float): Learning rate
            bidirectional (bool): Whether to use bidirectional LSTM
            device (str): Device to use for computation ('cuda' or 'cpu')
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create network
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        self.criterion = nn.BCELoss()
        
        # Store hyperparameters
        self.hyperparams = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'bidirectional': bidirectional
        }
        
        # For early stopping
        self.best_val_acc = 0
        self.patience_counter = 0
        self.max_patience = 10
        
        # Store training history
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        
        # For storing feature names
        self.feature_names = []
        
        # For calibration
        self.threshold = 0.5
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=50, 
            feature_names=None, class_weight=None, early_stopping=True):
        """
        Train the LSTM model
        
        Args:
            X_train (numpy.ndarray): Training features (batch_size, seq_len, input_size)
            y_train (numpy.ndarray): Training targets
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation targets
            batch_size (int): Batch size
            epochs (int): Number of epochs
            feature_names (list): List of feature names
            class_weight (dict): Class weights for imbalanced datasets
            early_stopping (bool): Whether to use early stopping
            
        Returns:
            self: Trained model
        """
        logger.info("Training LSTM model")
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Prepare data loaders
        train_loader = self._prepare_data_loader(X_train, y_train, batch_size, shuffle=True)
        
        # Validation data loader if provided
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._prepare_data_loader(X_val, y_val, batch_size, shuffle=False)
        
        # Handle class imbalance
        if class_weight is None:
            # Compute class weights if not provided
            unique, counts = np.unique(y_train, return_counts=True)
            if len(unique) == 2 and counts[0] != counts[1]:
                weight_for_0 = counts[1] / len(y_train)
                weight_for_1 = counts[0] / len(y_train)
                class_weight = {0: weight_for_0, 1: weight_for_1}
                logger.info(f"Class weights computed: {class_weight}")
        
        # Update loss function with class weights if provided
        if class_weight is not None:
            weights = torch.FloatTensor([class_weight[0], class_weight[1]]).to(self.device)
            self.criterion = nn.BCELoss(weight=weights[1])
            logger.info(f"Using weighted BCE loss with weights: {weights}")
        
        # Initialize best accuracy for early stopping
        best_val_acc = 0 if early_stopping else None
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            all_preds = []
            all_targets = []
            
            for batch_X, batch_y in train_loader:
                # Forward pass
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item()
                
                # Store predictions and targets
                predicted = (outputs.detach().cpu().numpy() > 0.5).astype(int)
                all_preds.extend(predicted)
                all_targets.extend(batch_y.cpu().numpy())
            
            # Calculate training metrics
            train_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(all_targets, all_preds)
            train_f1 = f1_score(all_targets, all_preds)
            
            # Validation phase
            val_loss = 0
            val_acc = 0
            val_f1 = 0
            
            if val_loader:
                val_loss, val_acc, val_f1 = self._evaluate(val_loader)
                
                # Learning rate scheduler based on validation accuracy
                self.scheduler.step(val_f1)
                
                # Early stopping
                if early_stopping:
                    if val_f1 > best_val_acc:
                        best_val_acc = val_f1
                        patience_counter = 0
                        # Save best model
                        self._save_checkpoint('best_model.pt')
                    else:
                        patience_counter += 1
                        if patience_counter >= self.max_patience:
                            logger.info(f"Early stopping at epoch {epoch+1}")
                            break
            
            # Log progress
            logger.info(f"Epoch [{epoch+1}/{epochs}], "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Store metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            if val_loader:
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        # Load best model if early stopping was used
        if early_stopping and os.path.exists('best_model.pt'):
            self._load_checkpoint('best_model.pt')
            os.remove('best_model.pt')  # Clean up
        
        return self
    
    def _prepare_data_loader(self, X, y, batch_size, shuffle=True):
        """
        Create PyTorch DataLoader from numpy arrays
        
        Args:
            X (numpy.ndarray): Features
            y (numpy.ndarray): Targets
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data
            
        Returns:
            DataLoader: PyTorch DataLoader
        """
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        
        return loader
    
    def _evaluate(self, data_loader):
        """
        Evaluate model on data loader
        
        Args:
            data_loader (DataLoader): PyTorch DataLoader
            
        Returns:
            tuple: (loss, accuracy, f1_score)
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X).squeeze()
                loss = self.criterion(outputs, batch_y)
                
                # Accumulate loss
                total_loss += loss.item()
                
                # Store predictions and targets
                predicted = (outputs.cpu().numpy() > 0.5).astype(int)
                all_preds.extend(predicted)
                all_targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        loss = total_loss / len(data_loader)
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        
        return loss, acc, f1
    
    def predict(self, X, threshold=None):
        """
        Make binary predictions
        
        Args:
            X (numpy.ndarray): Input features
            threshold (float): Custom probability threshold
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        # Use custom threshold if provided, otherwise use default or calibrated
        if threshold is None and hasattr(self, 'threshold'):
            threshold = self.threshold
        elif threshold is None:
            threshold = 0.5
        
        # Get probabilities
        probas = self.predict_proba(X)
        
        # Apply threshold
        predictions = (probas > threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probability of positive class
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predicted probabilities
        """
        # Convert to PyTorch tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
        
        # Convert to numpy array
        probas = outputs.cpu().numpy()
        
        return probas
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """
        Evaluate model on test data
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test targets
            batch_size (int): Batch size
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Create data loader
        test_loader = self._prepare_data_loader(X_test, y_test, batch_size, shuffle=False)
        
        # Evaluate
        loss, acc, f1 = self._evaluate(test_loader)
        
        # Get predictions for other metrics
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'loss': loss,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Log metrics
        logger.info(f"Test accuracy: {acc:.4f}")
        logger.info(f"Test precision: {precision:.4f}")
        logger.info(f"Test recall: {recall:.4f}")
        logger.info(f"Test F1 score: {f1:.4f}")
        logger.info(f"Test loss: {loss:.4f}")
        
        return metrics
    
    def calibrate_threshold(self, X_val, y_val, batch_size=32):
        """
        Find optimal probability threshold using validation data
        
        Args:
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation targets
            batch_size (int): Batch size
            
        Returns:
            float: Optimal threshold
        """
        logger.info("Calibrating probability threshold")
        
        # Get probabilities
        probas = self.predict_proba(X_val)
        
        # Try different thresholds
        thresholds = np.arange(0.05, 0.95, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (probas > threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")
        
        # Store the threshold
        self.threshold = best_threshold
        
        return best_threshold
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path (str): Path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history and self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        if 'val_acc' in self.history and self.history['val_acc']:
            plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(self.history['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        
        # Tighten layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.close()
    
    def explain_predictions(self, X, feature_names=None, max_display=10, plot=True, save_path=None):
        """
        Explain model predictions using SHAP values
        
        Args:
            X (numpy.ndarray): Input features
            feature_names (list): Feature names
            max_display (int): Maximum number of features to display
            plot (bool): Whether to plot the explanation
            save_path (str): Path to save the plot
            
        Returns:
            numpy.ndarray: SHAP values
        """
        # Use provided feature names or stored ones
        if feature_names is None:
            feature_names = self.feature_names if self.feature_names else None
        
        try:
            # Use a small sample for explanation if X is large
            if X.shape[0] > 100:
                sample_indices = np.random.choice(X.shape[0], 100, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X
            
            # Flatten sequences for SHAP - this is a simplification for explanation
            # In a real implementation, you would need more complex handling for time series
            if len(X_sample.shape) == 3:
                X_flat = X_sample.reshape(X_sample.shape[0], -1)
                
                # Adjust feature_names to reflect the flattened structure
                if feature_names:
                    flat_feature_names = []
                    for t in range(X_sample.shape[1]):  # For each time step
                        for f in feature_names:
                            flat_feature_names.append(f"{f}_t-{X_sample.shape[1]-t}")
                    feature_names = flat_feature_names
            else:
                X_flat = X_sample
            
            # Create a simple wrapper for the model to work with SHAP
            class ModelWrapper:
                def __init__(self, model):
                    self.model = model
                
                def __call__(self, X):
                    X_tensor = torch.FloatTensor(X).reshape(-1, X_sample.shape[1], X_sample.shape[2])
                    self.model.eval()
                    with torch.no_grad():
                        return self.model(X_tensor).cpu().numpy()
            
            # Create explainer
            explainer = shap.KernelExplainer(
                ModelWrapper(self.model),
                shap.sample(X_flat, min(50, X_flat.shape[0]))
            )
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_flat)
            
            # Plot summary
            if plot:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_values, 
                    X_flat, 
                    feature_names=feature_names,
                    max_display=max_display,
                    show=False
                )
                
                # Save if path provided
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path, bbox_inches='tight')
                    logger.info(f"SHAP summary plot saved to {save_path}")
                    plt.close()
                else:
                    plt.tight_layout()
                    plt.show()
            
            return shap_values
        
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {e}")
            return None
    
    def _save_checkpoint(self, filename):
        """
        Save model checkpoint
        
        Args:
            filename (str): Filename for the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hyperparams': self.hyperparams,
            'history': self.history,
            'threshold': self.threshold if hasattr(self, 'threshold') else 0.5,
            'feature_names': self.feature_names
        }
        torch.save(checkpoint, filename)
    
    def _load_checkpoint(self, filename):
        """
        Load model checkpoint
        
        Args:
            filename (str): Filename for the checkpoint
        """
        checkpoint = torch.load(filename)
        
        # Load model architecture and weights
        input_size = checkpoint['hyperparams']['input_size']
        hidden_size = checkpoint['hyperparams']['hidden_size']
        num_layers = checkpoint['hyperparams']['num_layers']
        dropout = checkpoint['hyperparams']['dropout']
        bidirectional = checkpoint['hyperparams']['bidirectional']
        
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        ).to(self.device)
        
        # Load weights and optimizer state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load other attributes
        self.history = checkpoint['history']
        self.threshold = checkpoint['threshold']
        self.feature_names = checkpoint['feature_names']
    
    def save(self, filepath):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self._save_checkpoint(filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, device=None):
        """
        Load model from file
        
        Args:
            filepath (str): Path to load model from
            device (str): Device to use (cuda or cpu)
            
        Returns:
            StockLSTMModel: Loaded model
        """
        # Set device
        device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location=device)
        
        # Get hyperparameters
        hyperparams = checkpoint['hyperparams']
        
        # Create model instance
        model_instance = cls(
            input_size=hyperparams['input_size'],
            hidden_size=hyperparams['hidden_size'],
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout'],
            learning_rate=hyperparams['learning_rate'],
            bidirectional=hyperparams['bidirectional'],
            device=device
        )
        
        # Load model weights
        model_instance.model.load_state_dict(checkpoint['model_state_dict'])
        model_instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model_instance.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load other attributes
        model_instance.history = checkpoint['history']
        model_instance.threshold = checkpoint['threshold']
        model_instance.feature_names = checkpoint['feature_names']
        
        logger.info(f"Model loaded from {filepath}")
        
        return model_instance