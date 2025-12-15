"""
Unit tests for prediction module.
"""
import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.predict import (
    load_model_and_encoders,
    prepare_prediction_features,
    predict_yield
)


class TestPrediction:
    """Test cases for prediction functions."""
    
    def test_prepare_prediction_features_basic(self):
        """Test basic feature preparation."""
        # Mock encoders and scaler
        class MockEncoder:
            def transform(self, x):
                return [0]
        
        encoders = {
            'crop': MockEncoder(),
            'district': MockEncoder()
        }
        
        # Test without scaler
        features = prepare_prediction_features(
            crop='Rice',
            district='Colombo',
            year=2024,
            rainfall=1500,
            temperature=27,
            humidity=75,
            encoders=encoders,
            scaler=None,
            feature_names=None,
            historical_yield=None
        )
        
        assert features is not None
        assert isinstance(features, np.ndarray)
    
    def test_prepare_prediction_features_with_historical(self):
        """Test feature preparation with historical yield."""
        class MockEncoder:
            def transform(self, x):
                return [0]
        
        encoders = {
            'crop': MockEncoder(),
            'district': MockEncoder()
        }
        
        features = prepare_prediction_features(
            crop='Rice',
            district='Colombo',
            year=2024,
            rainfall=1500,
            temperature=27,
            humidity=75,
            encoders=encoders,
            scaler=None,
            feature_names=None,
            historical_yield=1000.0
        )
        
        assert features is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



