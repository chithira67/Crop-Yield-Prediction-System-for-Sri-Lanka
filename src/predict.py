import pandas as pd
import numpy as np
import joblib
import os


def load_model_and_encoders(model_path='model/best_model.pkl'):
    """
    Load the best model, encoders, scaler, and feature names
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dictionary with model, encoders, scaler, and feature_names
    """
    try:
        model = joblib.load(model_path)
        
        encoders = {}
        if os.path.exists('model/crop_encoder.pkl'):
            encoders['crop'] = joblib.load('model/crop_encoder.pkl')
        if os.path.exists('model/district_encoder.pkl'):
            encoders['district'] = joblib.load('model/district_encoder.pkl')
        
        scaler = None
        if os.path.exists('model/scaler.pkl'):
            scaler = joblib.load('model/scaler.pkl')
        
        feature_names = None
        if os.path.exists('model/feature_names.pkl'):
            feature_names = joblib.load('model/feature_names.pkl')
        
        return {
            'model': model,
            'encoders': encoders,
            'scaler': scaler,
            'feature_names': feature_names
        }
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def prepare_prediction_features(crop, district, year, rainfall, temperature, 
                                humidity, encoders, scaler, feature_names,
                                historical_yield=None):
    """
    Prepare features for prediction matching the training data structure
    
    Args:
        crop: Crop name
        district: District name
        year: Year for prediction
        rainfall: Rainfall in mm
        temperature: Temperature in Celsius
        humidity: Humidity percentage
        encoders: Dictionary with crop and district encoders
        scaler: Fitted scaler
        feature_names: List of feature names in order
        historical_yield: Optional historical yield for lag features
        
    Returns:
        numpy array of features ready for prediction
    """
    try:
        # Load a sample from processed data to understand feature structure
        sample_df = None
        if os.path.exists('data/processed/features.csv'):
            sample_df = pd.read_csv('data/processed/features.csv', nrows=1)
        
        # Encode crop
        if 'crop' in encoders:
            try:
                crop_encoded = encoders['crop'].transform([crop])[0]
            except ValueError:
                crop_encoded = 0
        else:
            crop_encoded = 0
        
        # Encode district
        if 'district' in encoders:
            try:
                district_encoded = encoders['district'].transform([district])[0]
            except ValueError:
                district_encoded = 0
        else:
            district_encoded = 0
        
        # Calculate lag features
        yield_lag1 = historical_yield if historical_yield else 0
        yield_lag2 = 0
        yield_rolling_mean_3 = yield_lag1
        yield_rolling_std_3 = 0
        
        # Temporal features
        year_sin = np.sin(2 * np.pi * year / 10)
        year_cos = np.cos(2 * np.pi * year / 10)
        year_normalized = (year - 2019) / 4  # Assuming 2019-2023 range
        
        # Weather features
        rainfall_temp_interaction = rainfall * temperature
        temp_humidity_interaction = temperature * humidity
        
        # Create base feature dictionary
        base_features = {
            'year': year,
            'crop_encoded': crop_encoded,
            'district_encoded': district_encoded,
            'yield_lag1': yield_lag1,
            'yield_lag2': yield_lag2,
            'yield_rolling_mean_3': yield_rolling_mean_3,
            'yield_rolling_std_3': yield_rolling_std_3,
            'year_sin': year_sin,
            'year_cos': year_cos,
            'year_normalized': year_normalized,
            'rainfall': rainfall,
            'temperature': temperature,
            'humidity': humidity,
            'rainfall_temp_interaction': rainfall_temp_interaction,
            'temp_humidity_interaction': temp_humidity_interaction
        }
        
        # If we have a sample dataframe, use it to create matching structure
        if sample_df is not None and feature_names:
            # Create a row with base features
            feature_row = pd.DataFrame([base_features])
            
            # Add one-hot encoded crop columns (set to 0 for all, then 1 for matching crop)
            crop_cols = [col for col in feature_names if col.startswith('crop_')]
            for col in crop_cols:
                # Extract crop name from column name (e.g., 'crop_Rice' -> 'Rice')
                crop_name = col.replace('crop_', '')
                if crop_name.lower() in crop.lower() or crop.lower() in crop_name.lower():
                    feature_row[col] = 1
                else:
                    feature_row[col] = 0
            
            # Reorder to match feature_names (excluding 'yield', 'crop', 'district')
            feature_cols = [f for f in feature_names if f not in ['yield', 'crop', 'district']]
            
            # Ensure all columns exist (fill missing with 0)
            for col in feature_cols:
                if col not in feature_row.columns:
                    feature_row[col] = 0
            
            # Select and reorder columns
            feature_vector = feature_row[feature_cols].values
            
            # Scale if scaler is available
            if scaler is not None:
                feature_vector = scaler.transform(feature_vector)
            
            return feature_vector
        else:
            # Fallback: create feature vector from base features only
            # This won't work perfectly if one-hot encoding was used, but it's a fallback
            feature_vector = np.array([
                base_features['year'],
                base_features['crop_encoded'],
                base_features['district_encoded'],
                base_features['yield_lag1'],
                base_features['yield_lag2'],
                base_features['yield_rolling_mean_3'],
                base_features['yield_rolling_std_3'],
                base_features['year_sin'],
                base_features['year_cos'],
                base_features['year_normalized'],
                base_features['rainfall'],
                base_features['temperature'],
                base_features['humidity'],
                base_features['rainfall_temp_interaction'],
                base_features['temp_humidity_interaction']
            ]).reshape(1, -1)
            
            if scaler is not None:
                feature_vector = scaler.transform(feature_vector)
            
            return feature_vector
    
    except Exception as e:
        print(f"Error preparing features: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def predict_yield(crop, district, year, rainfall, temperature, humidity,
                  model_path='model/best_model.pkl', historical_yield=None):
    """
    Make a yield prediction
    
    Args:
        crop: Crop name
        district: District name
        year: Year for prediction
        rainfall: Rainfall in mm
        temperature: Temperature in Celsius
        humidity: Humidity percentage
        model_path: Path to model file
        historical_yield: Optional historical yield for lag features
        
    Returns:
        Predicted yield value
    """
    try:
        # Load model and encoders
        model_data = load_model_and_encoders(model_path)
        if model_data is None:
            return None
        
        # Prepare features
        features = prepare_prediction_features(
            crop, district, year, rainfall, temperature, humidity,
            model_data['encoders'], model_data['scaler'], 
            model_data['feature_names'], historical_yield
        )
        
        if features is None:
            return None
        
        # Make prediction
        prediction = model_data['model'].predict(features)[0]
        
        return prediction
    
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage
    prediction = predict_yield(
        crop='Rice',
        district='Colombo',
        year=2024,
        rainfall=1500,
        temperature=27,
        humidity=75
    )
    
    if prediction:
        print(f"Predicted Yield: {prediction:.2f}")
    else:
        print("Prediction failed")

