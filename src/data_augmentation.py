"""
Data augmentation techniques for small datasets.
"""
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.preprocessing import StandardScaler


def augment_with_smote_regression(X: pd.DataFrame, y: pd.Series, 
                                  k_neighbors: int = 5,
                                  n_samples: Optional[int] = None) -> tuple:
    """
    Augment data using SMOTE for regression (synthetic minority oversampling)
    
    Args:
        X: Feature dataframe
        y: Target series
        k_neighbors: Number of neighbors for SMOTE
        n_samples: Number of samples to generate (default: same as original)
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    try:
        from imblearn.over_sampling import SMOTERegression
    except ImportError:
        print("Warning: imbalanced-learn not installed. Using simple augmentation.")
        return augment_with_noise(X, y, n_samples=n_samples)
    
    if n_samples is None:
        n_samples = len(X)
    
    smote = SMOTERegression(k_neighbors=min(k_neighbors, len(X) - 1), random_state=42)
    X_aug, y_aug = smote.fit_resample(X, y)
    
    # Limit to n_samples if specified
    if len(X_aug) > n_samples:
        indices = np.random.choice(len(X_aug), n_samples, replace=False)
        X_aug = X_aug.iloc[indices] if isinstance(X_aug, pd.DataFrame) else X_aug[indices]
        y_aug = y_aug.iloc[indices] if isinstance(y_aug, pd.Series) else y_aug[indices]
    
    return X_aug, y_aug


def augment_with_noise(X: pd.DataFrame, y: pd.Series,
                       noise_level: float = 0.05,
                       n_samples: Optional[int] = None) -> tuple:
    """
    Augment data by adding Gaussian noise
    
    Args:
        X: Feature dataframe
        y: Target series
        noise_level: Standard deviation of noise (as fraction of feature std)
        n_samples: Number of samples to generate
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    if n_samples is None:
        n_samples = len(X)
    
    # Calculate noise scale
    X_std = X.std()
    noise_scale = X_std * noise_level
    
    # Generate augmented samples
    X_aug_list = []
    y_aug_list = []
    
    for _ in range(n_samples):
        # Randomly select a base sample
        idx = np.random.randint(0, len(X))
        base_X = X.iloc[idx].copy()
        base_y = y.iloc[idx]
        
        # Add noise
        noise = np.random.normal(0, noise_scale, size=len(base_X))
        augmented_X = base_X + noise
        
        # Add small noise to target
        y_noise = np.random.normal(0, y.std() * noise_level)
        augmented_y = base_y + y_noise
        
        X_aug_list.append(augmented_X)
        y_aug_list.append(augmented_y)
    
    X_aug = pd.DataFrame(X_aug_list, columns=X.columns)
    y_aug = pd.Series(y_aug_list)
    
    return X_aug, y_aug


def augment_with_interpolation(X: pd.DataFrame, y: pd.Series,
                               n_samples: Optional[int] = None,
                               alpha: float = 0.5) -> tuple:
    """
    Augment data by interpolating between samples
    
    Args:
        X: Feature dataframe
        y: Target series
        n_samples: Number of samples to generate
        alpha: Interpolation factor (0-1)
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    if n_samples is None:
        n_samples = len(X)
    
    X_aug_list = []
    y_aug_list = []
    
    for _ in range(n_samples):
        # Randomly select two samples
        idx1, idx2 = np.random.choice(len(X), 2, replace=False)
        
        # Interpolate
        X_interp = X.iloc[idx1] * alpha + X.iloc[idx2] * (1 - alpha)
        y_interp = y.iloc[idx1] * alpha + y.iloc[idx2] * (1 - alpha)
        
        X_aug_list.append(X_interp)
        y_aug_list.append(y_interp)
    
    X_aug = pd.DataFrame(X_aug_list, columns=X.columns)
    y_aug = pd.Series(y_aug_list)
    
    return X_aug, y_aug


def augment_data(X: pd.DataFrame, y: pd.Series,
                method: str = 'noise',
                n_samples: Optional[int] = None,
                **kwargs) -> tuple:
    """
    Augment data using specified method
    
    Args:
        X: Feature dataframe
        y: Target series
        method: Augmentation method ('noise', 'interpolation', 'smote')
        n_samples: Number of samples to generate
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Tuple of (augmented_X, augmented_y)
    """
    print(f"\nAugmenting data using {method} method...")
    print(f"Original samples: {len(X)}")
    
    if method == 'noise':
        X_aug, y_aug = augment_with_noise(X, y, n_samples=n_samples, **kwargs)
    elif method == 'interpolation':
        X_aug, y_aug = augment_with_interpolation(X, y, n_samples=n_samples, **kwargs)
    elif method == 'smote':
        X_aug, y_aug = augment_with_smote_regression(X, y, n_samples=n_samples, **kwargs)
    else:
        raise ValueError(f"Unknown augmentation method: {method}")
    
    print(f"Augmented samples: {len(X_aug)}")
    
    # Combine original and augmented
    X_combined = pd.concat([X, X_aug], ignore_index=True)
    y_combined = pd.concat([y, y_aug], ignore_index=True)
    
    print(f"Total samples after augmentation: {len(X_combined)}")
    
    return X_combined, y_combined



