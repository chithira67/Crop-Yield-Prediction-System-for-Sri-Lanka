# 🌾 Crop Yield Prediction System for Sri Lanka

A comprehensive, industry-ready machine learning system for predicting crop yields in Sri Lanka using advanced ML algorithms and interactive visualizations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a complete machine learning pipeline for predicting crop yields in Sri Lanka. The system uses multiple ML algorithms (Random Forest, XGBoost, LightGBM) with hyperparameter tuning to provide accurate yield predictions based on:

- **Crop type** (Rice, Tea, Coconut, Spices, etc.)
- **District location** (25 districts of Sri Lanka)
- **Weather conditions** (Rainfall, Temperature, Humidity)
- **Historical trends** (Lag features, rolling statistics)
- **Temporal features** (Year, seasonal patterns)

## ✨ Features

### 🔬 Data Processing
- Automated data cleaning and preprocessing
- Missing value handling
- Feature engineering (lag features, seasonal features, interactions)
- Categorical encoding (Label encoding, One-hot encoding)
- Feature scaling and normalization

### 🤖 Machine Learning
- **Three ML Models**: Random Forest, XGBoost, LightGBM
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV
- **Model Evaluation**: RMSE and R² metrics
- **Model Comparison**: Automatic selection of best-performing model
- **Feature Importance**: Analysis of most important features

### 📊 Visualization
- Interactive EDA notebooks with comprehensive charts
- Yearly yield trends
- Crop-wise comparisons
- Correlation matrices
- Feature importance visualizations
- District-level yield maps

### 🎨 Streamlit Dashboard
- **Interactive UI**: User-friendly interface for predictions
- **Real-time Predictions**: Get instant yield predictions
- **Analytics Dashboard**: Historical trends and patterns
- **Interactive Maps**: Folium-based district-level visualizations
- **Model Comparison**: Side-by-side model performance

## 📁 Project Structure

```
Crop-Yield-Prediction-System-for-Sri-Lanka/
│
├── data/
│   ├── raw/
│   │   └── crop_yield_data.csv          # Raw dataset
│   └── processed/
│       └── features.csv                  # Processed features
│
├── src/
│   ├── data_processing.py               # Data cleaning & feature engineering
│   ├── train_models.py                  # ML model training
│   ├── app.py                           # Streamlit dashboard
│   └── map_utils.py                     # Map visualization utilities
│
├── notebooks/
│   ├── 01_EDA.ipynb                     # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb     # Feature engineering demo
│   └── 03_Model_Training.ipynb          # Model training demo
│
├── model/
│   ├── best_model.pkl                   # Best trained model
│   ├── random_forest_model.pkl          # Random Forest model
│   ├── xgboost_model.pkl                 # XGBoost model
│   ├── lightgbm_model.pkl               # LightGBM model
│   ├── crop_encoder.pkl                  # Crop label encoder
│   ├── district_encoder.pkl              # District label encoder
│   ├── scaler.pkl                        # Feature scaler
│   └── feature_names.pkl                 # Feature names
│
├── images/                              # Generated visualizations
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── .gitignore                           # Git ignore rules
```

## 📊 Dataset

The dataset (`crop_yield_data.csv`) contains crop yield information for Sri Lanka with the following structure:

- **Area**: District/Location (Sri Lanka)
- **Item**: Crop type (Rice, Tea, Coconut, Spices, etc.)
- **Year**: Year of data (2019-2023)
- **Element**: Type of measurement (Yield, Production, Area harvested)
- **Value**: Yield value

### Dataset Statistics
- **Total Records**: ~2,660 rows
- **Unique Crops**: Multiple crop types
- **Time Period**: 2019-2023
- **Target Variable**: Yield (Element = "Yield")

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Crop-Yield-Prediction-System-for-Sri-Lanka.git
cd Crop-Yield-Prediction-System-for-Sri-Lanka
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Data Processing

Process the raw data to create features:

```bash
python src/data_processing.py
```

This will:
- Load and clean the raw data
- Create lag features, seasonal features, and weather features
- Encode categorical variables
- Scale numeric features
- Save processed data to `data/processed/features.csv`

### 2. Model Training

Train all ML models:

```bash
python src/train_models.py
```

This will:
- Load processed features
- Split data into train/test sets
- Train Random Forest, XGBoost, and LightGBM models
- Perform hyperparameter tuning
- Evaluate models using RMSE and R²
- Save the best model and all trained models

### 3. Run Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run src/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 4. Jupyter Notebooks

For step-by-step exploration:

```bash
jupyter notebook notebooks/
```

Then open:
- `01_EDA.ipynb` - Exploratory Data Analysis
- `02_Feature_Engineering.ipynb` - Feature engineering process
- `03_Model_Training.ipynb` - Model training and evaluation

## 🤖 Models

### Random Forest
- Ensemble method using multiple decision trees
- Handles non-linear relationships well
- Provides feature importance

### XGBoost
- Gradient boosting framework
- Excellent performance on structured data
- Built-in regularization

### LightGBM
- Fast gradient boosting framework
- Efficient memory usage
- Good for large datasets

### Model Selection
The system automatically selects the best model based on **lowest RMSE** on the test set.

### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**: Lower is better
- **R² (Coefficient of Determination)**: Higher is better (0-1 scale)

## 🎨 Streamlit Dashboard

The Streamlit dashboard provides:

### 🏠 Home Page
- Project overview
- Key statistics
- Quick navigation

### 🔮 Predict Yield
- Interactive form to input:
  - Crop type
  - District
  - Year
  - Weather conditions (Rainfall, Temperature, Humidity)
- Real-time predictions from all models
- Best model recommendation

### 📈 Analytics
- Yearly yield trends
- Top crops by yield
- Feature importance charts
- Statistical summaries

### 🗺️ Map View
- Interactive Folium map
- District-level yield visualization
- Color-coded markers based on yield
- District statistics table

## 📈 Results

### Model Performance
After training, the models are evaluated and compared. The best model is automatically selected and saved.

### Feature Importance
The system identifies the most important features for yield prediction, which may include:
- Historical yield (lag features)
- Weather conditions (rainfall, temperature, humidity)
- Crop type
- Temporal features (year, seasonal patterns)

## 🔧 Configuration

### Adjusting Hyperparameters
Edit `src/train_models.py` to modify hyperparameter search spaces:

```python
# Example: Random Forest parameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    # ... add more parameters
}
```

### Adding New Features
Modify `src/data_processing.py` to add new features:

```python
def create_custom_features(df):
    # Add your feature engineering logic
    return df
```

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your repository
4. Deploy!

### Local Deployment
```bash
streamlit run src/app.py --server.port 8501
```

## 📝 Notes

- The dataset uses "Sri Lanka" as the area. For district-level predictions, the system uses district inputs from the UI.
- Weather features are synthetic/placeholder. In production, integrate with weather APIs.
- Lag features require historical data. For new crops, these will be set to default values.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Chithira Jayarathna

## 🙏 Acknowledgments

- Sri Lanka Department of Agriculture for data
- Open-source ML community
- Streamlit team for the amazing framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for Sri Lankan Agriculture**

