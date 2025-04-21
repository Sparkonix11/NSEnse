import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

logger = logging.getLogger(__name__)

class StockXGBoostModel:
    """XGBoost model for stock movement prediction"""
    
    def __init__(self, params=None):
        """
        Initialize the XGBoost model
        
        Args:
            params (dict): XGBoost parameters
        """
        # Default parameters
        self.default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        # Use provided parameters or defaults
        self.params = params if params is not None else self.default_params
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        # For storing feature names
        self.feature_names = []
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """
        Train the XGBoost model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
            X_val (numpy.ndarray, optional): Validation features
            y_val (numpy.ndarray, optional): Validation targets
            feature_names (list, optional): Feature names for importance
        
        Returns:
            self: The trained model instance
        """
        logger.info("Training XGBoost model")
        
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Set up validation data if provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train the model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        # Log training results
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Validation metrics if available
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make binary predictions
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Make probability predictions
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Probability predictions
        """
        return self.model.predict_proba(X)
    
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
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
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
    
    def get_feature_importance(self):
        """
        Get feature importance
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create feature names if not provided
        if not self.feature_names:
            self.feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath):
        """
        Save model to file
        
        Args:
            filepath (str): Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'params': self.params,
                'feature_names': self.feature_names
            }, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load model from file
        
        Args:
            filepath (str): Path to load model from
            
        Returns:
            StockXGBoostModel: Loaded model
        """
        # Load model
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new model instance
        model_instance = cls(params=data['params'])
        
        # Set model and feature names
        model_instance.model = data['model']
        model_instance.feature_names = data['feature_names']
        
        logger.info(f"Model loaded from {filepath}")
        
        return model_instance