import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

logger = logging.getLogger(__name__)

class StockXGBoostModel:
    """XGBoost model for stock movement prediction"""
    
    def __init__(self, params=None):
        """
        Initialize the XGBoost model
        
        Args:
            params (dict): XGBoost parameters
        """
        # Default parameters - improved for better accuracy
        self.default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'random_state': 42,
            'eval_metric': 'logloss',
            'scale_pos_weight': 1,  # Adjust for class imbalance
            'min_child_weight': 3,
            'gamma': 0.1,  # Minimum loss reduction for partition
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 1.0,  # L2 regularization
            'max_delta_step': 0,
            'tree_method': 'hist',  # Faster algorithm
            'grow_policy': 'depthwise',
            'booster': 'gbtree',
            'importance_type': 'gain'
        }
        
        # Use provided parameters or defaults
        self.params = params if params is not None else self.default_params
        
        # Initialize model
        self.model = xgb.XGBClassifier(**self.params)
        
        # For storing feature names
        self.feature_names = []
        
        # For storing additional stats
        self.best_iteration = None
        self.cv_results = None

    def hyperparameter_tune(self, X_train, y_train, cv=3, n_iter=20):
        """
        Perform hyperparameter tuning using RandomizedSearchCV
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
            cv (int): Number of cross-validation folds
            n_iter (int): Number of parameter settings to try
        
        Returns:
            self: The model instance with optimized parameters
        """
        logger.info("Starting hyperparameter tuning")
        
        # Define the parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3, 0.5],
            'reg_alpha': [0, 0.1, 0.2, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0],
            'scale_pos_weight': [1, 2, 3]
        }
        
        # Use TimeSeriesSplit for time series data
        time_series_cv = TimeSeriesSplit(n_splits=cv)
        
        # Set up RandomizedSearchCV
        search = RandomizedSearchCV(
            xgb.XGBClassifier(objective='binary:logistic'),
            param_grid,
            n_iter=n_iter,
            cv=time_series_cv,
            scoring='accuracy',
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        # Fit the random search
        search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = search.best_params_
        best_score = search.best_score_
        
        # Update model parameters with best found parameters
        self.params = {**self.default_params, **best_params}
        
        # Reinitialize model with best parameters
        self.model = xgb.XGBClassifier(**self.params)
        
        logger.info(f"Best hyperparameters found: {best_params}")
        logger.info(f"Best CV score: {best_score:.4f}")
        
        # Store CV results for later analysis
        self.cv_results = search.cv_results_
        
        return self
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, tune_hyperparams=False):
        """
        Train the XGBoost model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training targets
            X_val (numpy.ndarray, optional): Validation features
            y_val (numpy.ndarray, optional): Validation targets
            feature_names (list, optional): Feature names for importance
            tune_hyperparams (bool): Whether to perform hyperparameter tuning
        
        Returns:
            self: The trained model instance
        """
        logger.info("Training XGBoost model")
        
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Hyperparameter tuning if requested
        if tune_hyperparams:
            self.hyperparameter_tune(X_train, y_train)
        
        # Check for class imbalance and adjust weights if needed
        unique, counts = np.unique(y_train, return_counts=True)
        if len(unique) == 2:
            ratio = max(counts) / min(counts)
            if ratio > 1.5:  # If imbalance detected
                logger.info(f"Class imbalance detected (ratio: {ratio:.2f}). Adjusting scale_pos_weight.")
                # Find minority class
                min_class = unique[np.argmin(counts)]
                # Set weight for minority class
                if min_class == 1:
                    self.params['scale_pos_weight'] = ratio
                    self.model.set_params(scale_pos_weight=ratio)
                else:
                    # If 0 is the minority class, invert the labels for XGBoost to handle properly
                    # or adjust through sample_weight
                    sample_weight = np.ones(len(y_train))
                    sample_weight[y_train == min_class] = ratio
                    self.model.set_params(scale_pos_weight=1)
        
        # Set up validation data if provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train the model with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True,
            early_stopping_rounds=50  # Stop if no improvement after 50 rounds
        )
        
        # Store best iteration
        if hasattr(self.model, 'best_iteration'):
            self.best_iteration = self.model.best_iteration
        
        # Log training results
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        
        # Calculate other metrics
        train_precision = precision_score(y_train, train_pred)
        train_recall = recall_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred)
        
        logger.info(f"Training precision: {train_precision:.4f}")
        logger.info(f"Training recall: {train_recall:.4f}")
        logger.info(f"Training F1: {train_f1:.4f}")
        
        # Validation metrics if available
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_precision = precision_score(y_val, val_pred)
            val_recall = recall_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred)
            
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
            logger.info(f"Validation precision: {val_precision:.4f}")
            logger.info(f"Validation recall: {val_recall:.4f}")
            logger.info(f"Validation F1: {val_f1:.4f}")
        
        return self
    
    def predict(self, X, threshold=None):
        """
        Make binary predictions with optional custom threshold
        
        Args:
            X (numpy.ndarray): Input features
            threshold (float, optional): Custom probability threshold
            
        Returns:
            numpy.ndarray: Binary predictions (0 or 1)
        """
        # Use calibrated threshold if available and no threshold provided
        if threshold is None and hasattr(self, 'threshold'):
            threshold = self.threshold
        elif threshold is None:
            threshold = 0.5
        
        # Get probability predictions
        proba = self.predict_proba(X)[:, 1]
        
        # Apply threshold
        predictions = (proba >= threshold).astype(int)
        
        return predictions
    
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
    
    def calibrate_threshold(self, X_val, y_val):
        """
        Find optimal probability threshold for predictions
        
        Args:
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation targets
        
        Returns:
            float: Optimal threshold value
        """
        logger.info("Calibrating probability threshold")
        
        # Get predictions
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Try different thresholds
        thresholds = np.arange(0.05, 0.95, 0.05)
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")
        
        # Store the threshold
        self.threshold = best_threshold
        
        return best_threshold
    
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
                'feature_names': self.feature_names,
                'best_iteration': self.best_iteration,
                'threshold': self.threshold if hasattr(self, 'threshold') else 0.5
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
        
        # Set additional attributes if available
        if 'best_iteration' in data:
            model_instance.best_iteration = data['best_iteration']
        
        if 'threshold' in data:
            model_instance.threshold = data['threshold']
        
        logger.info(f"Model loaded from {filepath}")
        
        return model_instance