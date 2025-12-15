import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os


def load_raw_data(file_path='data/raw/crop_yield_data.csv'):
    """
    Load raw crop yield data from CSV file
    
    Args:
        file_path: Path to the raw data CSV file
        
    Returns:
        DataFrame with raw data
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df


def clean_data(df):
    """
    Clean the raw dataset by handling missing values and filtering relevant data
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Filter for Yield data only (our target variable)
    df_clean = df[df['Element'] == 'Yield'].copy()
    
    # Remove rows with missing yield values
    df_clean = df_clean[df_clean['Value'].notna()].copy()
    
    # Select relevant columns
    df_clean = df_clean[['Area', 'Item', 'Year', 'Value']].copy()
    df_clean.columns = ['district', 'crop', 'year', 'yield']
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['district', 'crop', 'year'])
    
    # Filter out crops with insufficient data (less than 3 years)
    crop_counts = df_clean['crop'].value_counts()
    valid_crops = crop_counts[crop_counts >= 3].index
    df_clean = df_clean[df_clean['crop'].isin(valid_crops)].copy()
    
    print(f"Cleaned data: {len(df_clean)} rows, {df_clean['crop'].nunique()} crops")
    return df_clean


def create_lag_features(df):
    """
    Create lag features for yield (previous year's yield)
    
    Args:
        df: DataFrame with district, crop, year, yield columns
        
    Returns:
        DataFrame with lag features added
    """
    df = df.sort_values(['crop', 'year']).copy()
    
    # Lag features (previous year yield)
    df['yield_lag1'] = df.groupby('crop')['yield'].shift(1)
    df['yield_lag2'] = df.groupby('crop')['yield'].shift(2)
    
    # Rolling statistics
    df['yield_rolling_mean_3'] = df.groupby('crop')['yield'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['yield_rolling_std_3'] = df.groupby('crop')['yield'].transform(
        lambda x: x.rolling(window=3, min_periods=1).std()
    )
    
    return df


def create_seasonal_features(df):
    """
    Create seasonal and temporal features
    
    Args:
        df: DataFrame with year column
        
    Returns:
        DataFrame with seasonal features added
    """
    # Year features
    df['year_sin'] = np.sin(2 * np.pi * df['year'] / 10)
    df['year_cos'] = np.cos(2 * np.pi * df['year'] / 10)
    
    # Year normalized (for trend)
    min_year = df['year'].min()
    max_year = df['year'].max()
    df['year_normalized'] = (df['year'] - min_year) / (max_year - min_year) if max_year > min_year else 0
    
    return df


def encode_categorical_features(df):
    """
    Encode categorical variables (crop, district)
    
    Args:
        df: DataFrame with categorical columns
        
    Returns:
        DataFrame with encoded features and encoders
    """
    # Label encoding for crops
    crop_encoder = LabelEncoder()
    df['crop_encoded'] = crop_encoder.fit_transform(df['crop'])
    
    # Label encoding for districts
    district_encoder = LabelEncoder()
    df['district_encoded'] = district_encoder.fit_transform(df['district'])
    
    # One-hot encoding for crops (alternative approach)
    crop_dummies = pd.get_dummies(df['crop'], prefix='crop')
    df = pd.concat([df, crop_dummies], axis=1)
    
    return df, crop_encoder, district_encoder


def add_weather_features(df, use_api: bool = False, api_key: str = None):
    """
    Add weather features using real API or historical estimates
    
    Args:
        df: DataFrame
        use_api: Whether to use OpenWeatherMap API (requires API key)
        api_key: OpenWeatherMap API key (optional, can use env var)
        
    Returns:
        DataFrame with weather features
    """
    try:
        from src.weather_api import add_weather_features_real
        # Use real weather API integration
        df = add_weather_features_real(df, use_api=use_api, api_key=api_key)
    except ImportError:
        # Fallback to synthetic data if weather_api not available
        import numpy as np
        np.random.seed(42)
        df['rainfall'] = np.random.normal(1500, 300, len(df)).clip(800, 2500)
        df['temperature'] = np.random.normal(27, 2, len(df)).clip(22, 32)
        df['humidity'] = np.random.normal(75, 5, len(df)).clip(60, 90)
        df['rainfall_temp_interaction'] = df['rainfall'] * df['temperature']
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    
    return df


def scale_numeric_features(df, numeric_cols, scaler=None, fit=True):
    """
    Scale numeric features using StandardScaler
    
    Args:
        df: DataFrame
        numeric_cols: List of numeric column names to scale
        scaler: Pre-fitted scaler (if available)
        fit: Whether to fit the scaler
        
    Returns:
        DataFrame with scaled features and scaler object
    """
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df, scaler


def process_data(input_path='data/raw/crop_yield_data.csv', 
                 output_path='data/processed/features.csv',
                 save_encoders=True,
                 use_weather_api=False,
                 weather_api_key=None):
    """
    Main function to process raw data through entire pipeline
    
    Args:
        input_path: Path to raw data CSV
        output_path: Path to save processed features
        save_encoders: Whether to save encoders for later use
        
    Returns:
        Processed DataFrame, encoders, and scaler
    """
    print("=" * 50)
    print("Starting Data Processing Pipeline")
    print("=" * 50)
    
    # Load data
    df = load_raw_data(input_path)
    
    # Clean data
    df = clean_data(df)
    
    # Create lag features
    df = create_lag_features(df)
    
    # Create seasonal features
    df = create_seasonal_features(df)
    
    # Encode categorical features
    df, crop_encoder, district_encoder = encode_categorical_features(df)
    
    # Add weather features (using real API or historical estimates)
    df = add_weather_features(df, use_api=use_weather_api, api_key=weather_api_key)
    
    # Fill NaN values in lag features with mean
    lag_cols = ['yield_lag1', 'yield_lag2', 'yield_rolling_std_3']
    for col in lag_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    # Select features for model (exclude target and original categorical)
    feature_cols = [col for col in df.columns 
                    if col not in ['yield', 'crop', 'district']]
    
    # Scale numeric features (excluding encoded and one-hot encoded)
    numeric_cols = ['year', 'yield_lag1', 'yield_lag2', 'yield_rolling_mean_3', 
                   'yield_rolling_std_3', 'year_sin', 'year_cos', 'year_normalized',
                   'rainfall', 'temperature', 'humidity', 'rainfall_temp_interaction',
                   'temp_humidity_interaction']
    
    # Only scale columns that exist
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    df_scaled, scaler = scale_numeric_features(df, numeric_cols, fit=True)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")
    print(f"Final shape: {df_scaled.shape}")
    print(f"Features: {len(feature_cols)}")
    
    # Save encoders and scaler
    if save_encoders:
        os.makedirs('model', exist_ok=True)
        joblib.dump(crop_encoder, 'model/crop_encoder.pkl')
        joblib.dump(district_encoder, 'model/district_encoder.pkl')
        joblib.dump(scaler, 'model/scaler.pkl')
        print("Encoders and scaler saved to model/ directory")
    
    print("=" * 50)
    print("Data Processing Complete!")
    print("=" * 50)
    
    return df_scaled, crop_encoder, district_encoder, scaler


if __name__ == "__main__":
    # Run data processing pipeline
    processed_df, crop_enc, district_enc, scaler_obj = process_data()
    
    print("\nSample of processed data:")
    print(processed_df.head())
    print("\nData summary:")
    print(processed_df.describe())

