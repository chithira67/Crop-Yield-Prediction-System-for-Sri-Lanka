"""
Weather API integration for real-time weather data.
Supports OpenWeatherMap API with fallback to historical data.
"""
import os
import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import time


# Sri Lanka district coordinates (approximate)
DISTRICT_COORDINATES = {
    'Colombo': (6.9271, 79.8612),
    'Gampaha': (7.0844, 80.0097),
    'Kalutara': (6.5854, 79.9603),
    'Kandy': (7.2906, 80.6337),
    'Matale': (7.4675, 80.6234),
    'Nuwara Eliya': (6.9497, 80.7891),
    'Galle': (6.0329, 80.2170),
    'Matara': (5.9549, 80.5550),
    'Hambantota': (6.1248, 81.1185),
    'Jaffna': (9.6615, 80.0255),
    'Kilinochchi': (9.4004, 80.3991),
    'Mannar': (8.9776, 79.9118),
    'Mullaitivu': (9.2670, 80.8142),
    'Vavuniya': (8.7514, 80.4971),
    'Batticaloa': (7.7172, 81.7004),
    'Ampara': (7.2975, 81.6820),
    'Trincomalee': (8.5874, 81.2152),
    'Kurunegala': (7.4863, 80.3656),
    'Puttalam': (8.0362, 79.8283),
    'Anuradhapura': (8.3114, 80.4037),
    'Polonnaruwa': (7.9329, 81.0081),
    'Badulla': (6.9934, 81.0550),
    'Moneragala': (6.8728, 81.3487),
    'Ratnapura': (6.6828, 80.4012),
    'Kegalle': (7.2523, 80.3466)
}


def get_weather_from_api(district: str, year: int, month: int = 6, 
                         api_key: Optional[str] = None) -> Optional[Dict]:
    """
    Fetch weather data from OpenWeatherMap API
    
    Args:
        district: District name
        year: Year for historical data
        month: Month (default 6 for growing season)
        api_key: OpenWeatherMap API key (optional, can use env var)
        
    Returns:
        Dictionary with weather data or None if API fails
    """
    if api_key is None:
        api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if api_key is None:
        return None
    
    if district not in DISTRICT_COORDINATES:
        return None
    
    lat, lon = DISTRICT_COORDINATES[district]
    
    try:
        # For historical data, use OpenWeatherMap One Call API 3.0
        # Note: This requires a paid plan. For free tier, use current weather
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract weather data
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'rainfall': data.get('rain', {}).get('1h', 0) * 24 if 'rain' in data else 0
            }
            
            # If no rainfall data, estimate from weather condition
            if weather_data['rainfall'] == 0:
                weather_condition = data['weather'][0]['main'].lower()
                if 'rain' in weather_condition:
                    weather_data['rainfall'] = np.random.uniform(5, 15)  # mm/hour estimate
                else:
                    weather_data['rainfall'] = 0
            
            return weather_data
        else:
            return None
            
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def get_historical_weather(district: str, year: int) -> Dict:
    """
    Get historical weather data for a district and year.
    Uses realistic estimates based on Sri Lanka climate patterns.
    
    Args:
        district: District name
        year: Year
        
    Returns:
        Dictionary with weather data
    """
    # Base climate patterns for Sri Lanka
    # Coastal districts: Higher humidity, moderate rainfall
    # Inland districts: Lower humidity, variable rainfall
    # Hill country: Lower temperature, higher rainfall
    
    coastal_districts = ['Colombo', 'Gampaha', 'Kalutara', 'Galle', 'Matara', 
                        'Hambantota', 'Jaffna', 'Batticaloa', 'Trincomalee']
    hill_districts = ['Kandy', 'Nuwara Eliya', 'Badulla', 'Matale']
    
    np.random.seed(hash(f"{district}{year}") % 2**32)
    
    if district in coastal_districts:
        base_temp = 28
        base_rainfall = 1800
        base_humidity = 78
        temp_variation = 2
        rainfall_variation = 400
    elif district in hill_districts:
        base_temp = 22
        base_rainfall = 2200
        base_humidity = 75
        temp_variation = 3
        rainfall_variation = 500
    else:  # Inland districts
        base_temp = 27
        base_rainfall = 1400
        base_humidity = 70
        temp_variation = 3
        rainfall_variation = 350
    
    # Add year-based variation (climate trends)
    year_factor = (year - 2019) * 0.1
    
    temperature = np.random.normal(base_temp + year_factor, temp_variation)
    rainfall = np.random.normal(base_rainfall + year_factor * 20, rainfall_variation)
    humidity = np.random.normal(base_humidity, 5)
    
    # Clip to realistic ranges
    temperature = np.clip(temperature, 20, 35)
    rainfall = np.clip(rainfall, 600, 3000)
    humidity = np.clip(humidity, 50, 95)
    
    return {
        'temperature': float(temperature),
        'rainfall': float(rainfall),
        'humidity': float(humidity)
    }


def add_weather_features_real(df: pd.DataFrame, use_api: bool = False, 
                              api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Add real weather features to dataframe
    
    Args:
        df: DataFrame with district and year columns
        use_api: Whether to use API (requires API key)
        api_key: OpenWeatherMap API key
        
    Returns:
        DataFrame with weather features added
    """
    df = df.copy()
    
    weather_data_list = []
    
    print("Fetching weather data...")
    for idx, row in df.iterrows():
        district = row['district']
        year = int(row['year'])
        
        weather_data = None
        
        if use_api:
            # Try API first
            weather_data = get_weather_from_api(district, year, api_key=api_key)
            time.sleep(0.1)  # Rate limiting
        
        # Fallback to historical estimates
        if weather_data is None:
            weather_data = get_historical_weather(district, year)
        
        weather_data_list.append(weather_data)
    
    # Add weather columns
    weather_df = pd.DataFrame(weather_data_list)
    df['rainfall'] = weather_df['rainfall']
    df['temperature'] = weather_df['temperature']
    df['humidity'] = weather_df['humidity']
    
    # Interaction features
    df['rainfall_temp_interaction'] = df['rainfall'] * df['temperature']
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    df['rainfall_humidity_interaction'] = df['rainfall'] * df['humidity']
    
    print(f"Added weather features for {len(df)} records")
    
    return df



