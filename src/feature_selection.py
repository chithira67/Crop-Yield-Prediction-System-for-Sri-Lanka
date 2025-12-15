"""
Feature selection utilities for improving model performance.
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Optional


def select_features_correlation(X: pd.DataFrame, y: pd.Series, 
                                 threshold: float = 0.01) -> List[str]:
    """
    Select features based on correlation with target
    
    Args:
        X: Feature dataframe
        y: Target series
        threshold: Minimum correlation threshold
        
    Returns:
        List of selected feature names
    """
    correlations = X.corrwith(y).abs()
    selected = correlations[correlations >= threshold].index.tolist()
    return selected


def select_features_univariate(X: pd.DataFrame, y: pd.Series, 
                              k: int = 20, score_func=f_regression) -> List[str]:
    """
    Select top k features using univariate statistical tests
    
    Args:
        X: Feature dataframe
        y: Target series
        k: Number of features to select
        score_func: Scoring function (f_regression or mutual_info_regression)
        
    Returns:
        List of selected feature names
    """
    selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features


def select_features_rfe(X: pd.DataFrame, y: pd.Series, 
                       n_features: int = 20, 
                       estimator: Optional[RandomForestRegressor] = None) -> List[str]:
    """
    Select features using Recursive Feature Elimination
    
    Args:
        X: Feature dataframe
        y: Target series
        n_features: Number of features to select
        estimator: Base estimator (default: RandomForestRegressor)
        
    Returns:
        List of selected feature names
    """
    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    
    selector = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]))
    selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features


def select_features_importance(X: pd.DataFrame, y: pd.Series,
                               threshold: Optional[float] = None,
                               max_features: Optional[int] = None) -> List[str]:
    """
    Select features based on Random Forest importance
    
    Args:
        X: Feature dataframe
        y: Target series
        threshold: Minimum importance threshold (percentile)
        max_features: Maximum number of features to select
        
    Returns:
        List of selected feature names
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    
    if threshold is not None:
        threshold_value = np.percentile(importances, threshold)
        selected = importances[importances >= threshold_value].index.tolist()
    else:
        selected = importances.index.tolist()
    
    if max_features is not None:
        selected = selected[:max_features]
    
    return selected


def select_features_combined(X: pd.DataFrame, y: pd.Series,
                            method: str = 'importance',
                            n_features: int = 20) -> Tuple[List[str], pd.DataFrame]:
    """
    Combined feature selection using multiple methods
    
    Args:
        X: Feature dataframe
        y: Target series
        method: Selection method ('importance', 'univariate', 'rfe', 'combined')
        n_features: Number of features to select
        
    Returns:
        Tuple of (selected feature names, feature importance dataframe)
    """
    print(f"\nFeature Selection using {method} method...")
    print(f"Original features: {X.shape[1]}")
    
    if method == 'importance':
        selected = select_features_importance(X, y, max_features=n_features)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[selected], y)
        importance_df = pd.DataFrame({
            'feature': selected,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
    elif method == 'univariate':
        selected = select_features_univariate(X, y, k=n_features)
        importance_df = pd.DataFrame({
            'feature': selected,
            'importance': [1.0] * len(selected)  # Placeholder
        })
        
    elif method == 'rfe':
        selected = select_features_rfe(X, y, n_features=n_features)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[selected], y)
        importance_df = pd.DataFrame({
            'feature': selected,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
    elif method == 'combined':
        # Use multiple methods and take intersection/union
        importance_features = set(select_features_importance(X, y, max_features=n_features))
        univariate_features = set(select_features_univariate(X, y, k=n_features))
        rfe_features = set(select_features_rfe(X, y, n_features=n_features))
        
        # Take features that appear in at least 2 methods
        all_features = importance_features | univariate_features | rfe_features
        feature_counts = {}
        for feature in all_features:
            count = sum([
                feature in importance_features,
                feature in univariate_features,
                feature in rfe_features
            ])
            feature_counts[feature] = count
        
        # Select top features by consensus
        selected = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:n_features]
        selected = [f[0] for f in selected]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X[selected], y)
        importance_df = pd.DataFrame({
            'feature': selected,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Selected features: {len(selected)}")
    
    return selected, importance_df



