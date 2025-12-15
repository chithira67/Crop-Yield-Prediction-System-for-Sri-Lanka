# Crop Yield Prediction System for Sri Lanka

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

A machine learning system for predicting crop yields in Sri Lanka using multiple ML algorithms, feature engineering, and interactive visualizations.

## Overview

This project implements an end-to-end machine learning pipeline for predicting crop yields in Sri Lanka. The system uses multiple ML algorithms (Random Forest, XGBoost, LightGBM) with hyperparameter tuning to provide accurate yield predictions based on:

- Crop type (Rice, Tea, Coconut, Spices, etc.)
- District location (25 districts of Sri Lanka)
- Weather conditions (Rainfall, Temperature, Humidity)
- Historical trends (Lag features, rolling statistics)
- Temporal features (Year, seasonal patterns)

## Features

### Core Features
- **Data Processing**: Automated cleaning, feature engineering, and preprocessing
- **Machine Learning**: Three models (Random Forest, XGBoost, LightGBM) with hyperparameter tuning
- **Evaluation**: RMSE and R² metrics with automatic best model selection
- **Visualization**: Interactive EDA notebooks and Streamlit dashboard
- **Deployment**: Streamlit dashboard for real-time predictions

### Advanced Features
- **Real Weather API Integration**: OpenWeatherMap API integration with historical weather data fallback
- **Feature Selection**: Multiple methods (importance-based, univariate, RFE, combined)
- **Time Series Cross-Validation**: Proper temporal validation for time-dependent data
- **Ensemble Stacking**: Meta-learner ensemble combining base models for improved performance
- **Model Explainability**: SHAP and LIME explanations for model interpretability
- **Data Augmentation**: Techniques for handling small datasets (noise, interpolation, SMOTE)

## Project Structure

```
Crop-Yield-Prediction-System-for-Sri-Lanka/
├── data/
│   ├── raw/                    # Raw dataset
│   └── processed/              # Processed features
├── src/                        # Source code
│   ├── config.py              # Configuration management
│   ├── logger.py              # Logging setup
│   ├── data_processing.py     # Data cleaning & feature engineering
│   ├── train_models.py        # ML model training (basic)
│   ├── train_models_enhanced.py  # Enhanced training with advanced features
│   ├── predict.py             # Prediction functions
│   ├── app.py                 # Streamlit dashboard
│   ├── map_utils.py           # Map visualization utilities
│   ├── weather_api.py         # Real weather API integration
│   ├── feature_selection.py  # Feature selection utilities
│   ├── time_series_cv.py      # Time series cross-validation
│   ├── ensemble_stacking.py   # Ensemble stacking implementation
│   ├── explainability.py      # SHAP/LIME model explanations
│   └── data_augmentation.py   # Data augmentation techniques
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks for EDA
├── model/                      # Trained models (generated)
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
└── run_pipeline.py             # Main pipeline script
```

## Installation

1. Clone the repository
   ```bash
   git clone https://github.com/yourusername/Crop-Yield-Prediction-System-for-Sri-Lanka.git
   cd Crop-Yield-Prediction-System-for-Sri-Lanka
   ```

2. Create virtual environment
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Processing

Process the raw data to create features:

```bash
python src/data_processing.py
```

Or use the complete pipeline:

```bash
python run_pipeline.py
```

### Model Training

**Basic Training:**
```bash
python src/train_models.py
```

**Enhanced Training** (with feature selection, time series CV, stacking, and explainability):
```bash
python src/train_models_enhanced.py
```

The enhanced training includes:
- Automatic feature selection
- Time series cross-validation
- Ensemble stacking
- SHAP model explanations

### Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run src/app.py
```

The dashboard will open at `http://localhost:8501`

### Make Predictions

```python
from src.predict import predict_yield

prediction = predict_yield(
    crop='Rice',
    district='Colombo',
    year=2024,
    rainfall=1500,
    temperature=27,
    humidity=75
)
print(f"Predicted Yield: {prediction:.2f}")
```

## Models

The system trains multiple models and automatically selects the best one:

- **Random Forest**: Best base model (RMSE: ~2787, R²: ~0.92)
- **XGBoost**: Performance (RMSE: ~3022, R²: ~0.90)
- **LightGBM**: Performance (RMSE: ~3030, R²: ~0.90)
- **Stacking Ensemble**: Meta-learner combining all base models (typically best overall performance)

### Advanced Techniques

1. **Feature Selection**: Automatically selects top 25 features using importance-based, univariate, or RFE methods
2. **Time Series Cross-Validation**: Proper temporal validation ensuring no data leakage
3. **Ensemble Stacking**: Combines base models with Ridge regression meta-learner
4. **Model Explainability**: SHAP values for understanding feature contributions

## Results & Impact

- Achieved **92% R² score** with Random Forest model
- Implemented comprehensive feature engineering pipeline (lag features, rolling statistics, seasonal encoding)
- Built production-ready Streamlit dashboard with interactive visualizations
- Automated model selection and hyperparameter tuning using cross-validation

## Configuration

Edit `config.yaml` to customize data paths, model parameters, and feature engineering settings.

## Testing

```bash
pytest tests/ -v
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Advanced Features Details

### Weather API Integration
- Real-time weather data from OpenWeatherMap API
- Automatic fallback to historical climate estimates
- District-specific weather patterns for Sri Lanka
- Set `OPENWEATHER_API_KEY` environment variable to use API

### Feature Selection
- Multiple selection methods: importance-based, univariate tests, RFE, combined
- Automatic selection of top N features
- Feature importance visualization

### Time Series Cross-Validation
- Proper temporal validation preventing data leakage
- Year-based splitting for time-dependent data
- Multiple CV folds for robust evaluation

### Ensemble Stacking
- Combines Random Forest, XGBoost, and LightGBM
- Ridge regression meta-learner
- Out-of-fold predictions for training meta-learner

### Model Explainability
- SHAP (SHapley Additive exPlanations) for global and local explanations
- LIME (Local Interpretable Model-agnostic Explanations) for instance-level explanations
- Feature importance plots and summary visualizations

### Data Augmentation
- Gaussian noise injection
- Interpolation between samples
- SMOTE for regression (when imbalanced-learn is available)

## Technical Skills Demonstrated

- **Machine Learning**: Random Forest, XGBoost, LightGBM, Ensemble Stacking, Hyperparameter Tuning
- **Feature Engineering**: Lag features, rolling statistics, temporal encoding, interaction features
- **Feature Selection**: Importance-based, univariate, RFE, combined methods
- **Time Series**: Time series cross-validation, temporal data handling
- **Model Explainability**: SHAP, LIME, feature importance analysis
- **Data Processing**: Data cleaning, missing value imputation, categorical encoding, feature scaling, data augmentation
- **API Integration**: OpenWeatherMap API, RESTful API design
- **Deployment**: Streamlit web application, model persistence, API design
- **Software Engineering**: Unit testing, configuration management, logging, code organization
- **Visualization**: Matplotlib, Seaborn, Folium maps, interactive dashboards

## Author

**Chithira Jayarathna**
