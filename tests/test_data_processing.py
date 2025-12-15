"""
Unit tests for data processing module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data_processing import (
    load_raw_data,
    clean_data,
    create_lag_features,
    create_seasonal_features,
    encode_categorical_features,
    add_weather_features,
    scale_numeric_features,
    process_data
)


class TestDataProcessing:
    """Test cases for data processing functions."""
    
    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data for testing."""
        return pd.DataFrame({
            'Area': ['Colombo', 'Kandy', 'Colombo', 'Kandy'] * 2,
            'Item': ['Rice', 'Rice', 'Tea', 'Tea'] * 2,
            'Year': [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022],
            'Element': ['Yield', 'Yield', 'Yield', 'Yield'] * 2,
            'Value': [1000.0, 1200.0, 800.0, 900.0, 1100.0, 1300.0, 850.0, 950.0]
        })
    
    @pytest.fixture
    def sample_cleaned_data(self):
        """Create sample cleaned data for testing."""
        return pd.DataFrame({
            'district': ['Colombo', 'Kandy', 'Colombo', 'Kandy'] * 2,
            'crop': ['Rice', 'Rice', 'Tea', 'Tea'] * 2,
            'year': [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022],
            'yield': [1000.0, 1200.0, 800.0, 900.0, 1100.0, 1300.0, 850.0, 950.0]
        })
    
    def test_clean_data(self, sample_raw_data):
        """Test data cleaning function."""
        cleaned = clean_data(sample_raw_data)
        
        assert 'district' in cleaned.columns
        assert 'crop' in cleaned.columns
        assert 'year' in cleaned.columns
        assert 'yield' in cleaned.columns
        assert len(cleaned) > 0
        assert cleaned['yield'].notna().all()
    
    def test_create_lag_features(self, sample_cleaned_data):
        """Test lag feature creation."""
        df_with_lags = create_lag_features(sample_cleaned_data)
        
        assert 'yield_lag1' in df_with_lags.columns
        assert 'yield_lag2' in df_with_lags.columns
        assert 'yield_rolling_mean_3' in df_with_lags.columns
        assert 'yield_rolling_std_3' in df_with_lags.columns
    
    def test_create_seasonal_features(self, sample_cleaned_data):
        """Test seasonal feature creation."""
        df_with_seasonal = create_seasonal_features(sample_cleaned_data)
        
        assert 'year_sin' in df_with_seasonal.columns
        assert 'year_cos' in df_with_seasonal.columns
        assert 'year_normalized' in df_with_seasonal.columns
    
    def test_encode_categorical_features(self, sample_cleaned_data):
        """Test categorical encoding."""
        df_encoded, crop_enc, district_enc = encode_categorical_features(sample_cleaned_data)
        
        assert 'crop_encoded' in df_encoded.columns
        assert 'district_encoded' in df_encoded.columns
        assert crop_enc is not None
        assert district_enc is not None
    
    def test_add_weather_features(self, sample_cleaned_data):
        """Test weather feature addition."""
        df_with_weather = add_weather_features(sample_cleaned_data)
        
        assert 'rainfall' in df_with_weather.columns
        assert 'temperature' in df_with_weather.columns
        assert 'humidity' in df_with_weather.columns
        assert 'rainfall_temp_interaction' in df_with_weather.columns
        assert 'temp_humidity_interaction' in df_with_weather.columns
        
        # Check value ranges (with some tolerance for edge cases)
        assert df_with_weather['rainfall'].between(500, 3000).all()
        assert df_with_weather['temperature'].between(20, 35).all()
        assert df_with_weather['humidity'].between(50, 100).all()
    
    def test_scale_numeric_features(self, sample_cleaned_data):
        """Test feature scaling."""
        numeric_cols = ['year', 'yield']
        df_scaled, scaler = scale_numeric_features(
            sample_cleaned_data, 
            numeric_cols, 
            fit=True
        )
        
        assert scaler is not None
        assert df_scaled[numeric_cols].notna().all().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



