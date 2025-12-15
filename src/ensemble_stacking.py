"""
Ensemble stacking for improved model performance.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Any
import joblib
import os


class StackingEnsemble:
    """
    Stacking ensemble that combines multiple base models with a meta-learner.
    """
    
    def __init__(self, base_models: Dict[str, Any], meta_model: Any = None,
                 use_probas: bool = False, cv_folds: int = 5):
        """
        Initialize stacking ensemble
        
        Args:
            base_models: Dictionary of base models {name: model}
            meta_model: Meta-learner model (default: Ridge regression)
            use_probas: Whether to use probabilities (for classification)
            cv_folds: Number of CV folds for generating meta-features
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.use_probas = use_probas
        self.cv_folds = cv_folds
        self.fitted_base_models = {}
        self.fitted_meta_model = None
    
    def _generate_meta_features(self, X: np.ndarray, y: np.ndarray,
                                X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate meta-features using out-of-fold predictions
        
        Args:
            X: Training features
            y: Training target
            X_test: Test features
            
        Returns:
            Tuple of (train_meta_features, test_meta_features)
        """
        from sklearn.model_selection import KFold
        
        n_train = X.shape[0]
        n_test = X_test.shape[0]
        n_models = len(self.base_models)
        
        train_meta_features = np.zeros((n_train, n_models))
        test_meta_features = np.zeros((n_test, n_models))
        
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Train each base model and generate predictions
        for model_idx, (model_name, model) in enumerate(self.base_models.items()):
            print(f"  Training {model_name} for stacking...")
            
            # Out-of-fold predictions for training set
            oof_predictions = np.zeros(n_train)
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Fit model on fold
                model_copy = self._clone_model(model)
                model_copy.fit(X_train_fold, y_train_fold)
                
                # Predict on validation fold
                oof_predictions[val_idx] = model_copy.predict(X_val_fold)
            
            train_meta_features[:, model_idx] = oof_predictions
            
            # Train on full training set and predict test
            model_full = self._clone_model(model)
            model_full.fit(X, y)
            self.fitted_base_models[model_name] = model_full
            
            test_meta_features[:, model_idx] = model_full.predict(X_test)
        
        return train_meta_features, test_meta_features
    
    def _clone_model(self, model: Any) -> Any:
        """Create a copy of the model"""
        import copy
        return copy.deepcopy(model)
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray = None):
        """
        Fit the stacking ensemble
        
        Args:
            X: Training features
            y: Training target
            X_test: Test features (optional, for generating test predictions)
        """
        print("Training Stacking Ensemble...")
        
        if X_test is None:
            # Use training data for meta-features (less ideal but simpler)
            X_test = X
        
        # Generate meta-features
        train_meta, test_meta = self._generate_meta_features(X, y, X_test)
        
        # Train meta-learner on meta-features
        print("  Training meta-learner...")
        self.fitted_meta_model = self._clone_model(self.meta_model)
        self.fitted_meta_model.fit(train_meta, y)
        
        print("Stacking ensemble trained!")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        # Generate meta-features using fitted base models
        n_samples = X.shape[0]
        n_models = len(self.fitted_base_models)
        meta_features = np.zeros((n_samples, n_models))
        
        for model_idx, (model_name, model) in enumerate(self.fitted_base_models.items()):
            meta_features[:, model_idx] = model.predict(X)
        
        # Predict using meta-learner
        predictions = self.fitted_meta_model.predict(meta_features)
        
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the stacking ensemble
        
        Args:
            X: Features
            y: True target values
            
        Returns:
            Dictionary with metrics
        """
        predictions = self.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)
        
        return {
            'RMSE': rmse,
            'R²': r2
        }
    
    def save(self, filepath: str):
        """Save the ensemble to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'base_models': self.fitted_base_models,
            'meta_model': self.fitted_meta_model,
            'base_model_configs': self.base_models
        }, filepath)
        print(f"Stacking ensemble saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load ensemble from disk"""
        data = joblib.load(filepath)
        ensemble = cls({}, meta_model=data['meta_model'])
        ensemble.fitted_base_models = data['base_models']
        ensemble.fitted_meta_model = data['meta_model']
        return ensemble



