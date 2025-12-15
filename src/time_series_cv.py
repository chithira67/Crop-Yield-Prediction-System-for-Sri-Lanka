"""
Time series cross-validation utilities.
"""
import numpy as np
import pandas as pd
from typing import Generator, Tuple, List
from sklearn.model_selection import BaseCrossValidator


class TimeSeriesSplit:
    """
    Time series cross-validator for temporal data.
    Ensures training data always comes before test data.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 1, gap: int = 0):
        """
        Initialize time series split
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set for each split
            gap: Gap between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set
        
        Args:
            X: Feature array
            y: Target array (optional)
            groups: Group labels (optional)
            
        Yields:
            Tuple of (train indices, test indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate split sizes
        test_size = self.test_size
        train_size = n_samples - (self.n_splits * test_size) - (self.n_splits * self.gap)
        
        if train_size <= 0:
            raise ValueError("Not enough samples for the specified number of splits")
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate start and end indices
            test_start = n_samples - (self.n_splits - i) * test_size - (self.n_splits - i) * self.gap
            test_end = test_start + test_size
            
            train_end = test_start - self.gap
            
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations"""
        return self.n_splits


class TimeSeriesSplitByYear:
    """
    Time series cross-validator that splits by year.
    Useful when data has a year column.
    """
    
    def __init__(self, n_splits: int = 3):
        """
        Initialize year-based time series split
        
        Args:
            n_splits: Number of splits (years to use for testing)
        """
        self.n_splits = n_splits
    
    def split(self, X: pd.DataFrame, y=None, year_col: str = 'year') -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices split by year
        
        Args:
            X: Feature dataframe (must have year column)
            y: Target array (optional)
            year_col: Name of year column
            
        Yields:
            Tuple of (train indices, test indices)
        """
        if year_col not in X.columns:
            raise ValueError(f"Year column '{year_col}' not found in dataframe")
        
        years = sorted(X[year_col].unique())
        
        if len(years) < self.n_splits + 1:
            raise ValueError(f"Not enough years ({len(years)}) for {self.n_splits} splits")
        
        # Use last n_splits years for testing
        test_years = years[-self.n_splits:]
        
        for test_year in test_years:
            train_indices = X[X[year_col] < test_year].index.values
            test_indices = X[X[year_col] == test_year].index.values
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations"""
        return self.n_splits


def create_time_series_splits(df: pd.DataFrame, year_col: str = 'year',
                               n_splits: int = 3) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time series splits for a dataframe with year column
    
    Args:
        df: DataFrame with year column
        year_col: Name of year column
        n_splits: Number of splits
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    splitter = TimeSeriesSplitByYear(n_splits=n_splits)
    splits = list(splitter.split(df, year_col=year_col))
    return splits



