import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import sys
sys.path.append(os.path.dirname(__file__))
from predict import predict_yield, load_model_and_encoders
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crop Yield Prediction - Sri Lanka",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4CAF50;
        text-align: center;
        padding: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed data and models"""
    try:
        # Load processed data
        if os.path.exists('data/processed/features.csv'):
            df = pd.read_csv('data/processed/features.csv')
        else:
            df = None
        
        # Load models
        models = {}
        model_files = {
            'best': 'model/best_model.pkl',
            'random_forest': 'model/random_forest_model.pkl',
            'xgboost': 'model/xgboost_model.pkl',
            'lightgbm': 'model/lightgbm_model.pkl'
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                models[name] = joblib.load(path)
        
        # Load encoders and scaler
        encoders = {}
        if os.path.exists('model/crop_encoder.pkl'):
            encoders['crop'] = joblib.load('model/crop_encoder.pkl')
        if os.path.exists('model/district_encoder.pkl'):
            encoders['district'] = joblib.load('model/district_encoder.pkl')
        if os.path.exists('model/scaler.pkl'):
            scaler = joblib.load('model/scaler.pkl')
        else:
            scaler = None
        
        # Load feature names
        if os.path.exists('model/feature_names.pkl'):
            feature_names = joblib.load('model/feature_names.pkl')
        else:
            feature_names = None
        
        return df, models, encoders, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, {}, {}, None, None


@st.cache_data
def load_raw_data():
    """Load raw data for visualizations"""
    try:
        if os.path.exists('data/raw/crop_yield_data.csv'):
            df = pd.read_csv('data/raw/crop_yield_data.csv')
            df_yield = df[df['Element'] == 'Yield'].copy()
            df_yield = df_yield[df_yield['Value'].notna()].copy()
            df_yield = df_yield[['Area', 'Item', 'Year', 'Value']].copy()
            df_yield.columns = ['district', 'crop', 'year', 'yield']
            return df_yield
        return None
    except:
        return None


def get_sri_lanka_districts():
    """Get list of Sri Lanka districts"""
    return [
        'Ampara', 'Anuradhapura', 'Badulla', 'Batticaloa', 'Colombo',
        'Galle', 'Gampaha', 'Hambantota', 'Jaffna', 'Kalutara',
        'Kandy', 'Kegalle', 'Kilinochchi', 'Kurunegala', 'Mannar',
        'Matale', 'Matara', 'Moneragala', 'Mullaitivu', 'Nuwara Eliya',
        'Polonnaruwa', 'Puttalam', 'Ratnapura', 'Trincomalee', 'Vavuniya'
    ]




def create_yield_map(df_yield):
    """Create a map visualization of yields by district"""
    # Sri Lanka center coordinates
    sri_lanka_center = [7.8731, 80.7718]
    
    # Create base map
    m = folium.Map(location=sri_lanka_center, zoom_start=7, tiles='OpenStreetMap')
    
    # Add markers for districts (simplified - would need actual coordinates)
    if df_yield is not None:
        district_yields = df_yield.groupby('district')['yield'].mean()
        for district, avg_yield in district_yields.items():
            # Placeholder coordinates - in real app, use actual district coordinates
            lat = sri_lanka_center[0] + np.random.uniform(-1, 1)
            lon = sri_lanka_center[1] + np.random.uniform(-1, 1)
            
            # Color based on yield
            if avg_yield > 50000:
                color = 'green'
            elif avg_yield > 20000:
                color = 'orange'
            else:
                color = 'red'
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                popup=f"{district}: {avg_yield:.0f}",
                color=color,
                fill=True,
                fillColor=color
            ).add_to(m)
    
    return m


def main():
    """Main Streamlit app"""
    # Header
    st.markdown('<h1 class="main-header">🌾 Crop Yield Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sri Lanka Agricultural Intelligence Platform</p>', unsafe_allow_html=True)
    
    # Load data and models
    df, models, encoders, scaler, feature_names = load_data()
    df_yield = load_raw_data()
    
    # Sidebar
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🔮 Predict Yield", "📈 Analytics", "🗺️ Map View"]
    )
    
    if page == "🏠 Home":
        st.markdown("## Welcome to the Crop Yield Prediction System")
        st.markdown("""
        This application uses machine learning to predict crop yields in Sri Lanka based on:
        - **Crop type**
        - **District location**
        - **Weather conditions** (rainfall, temperature, humidity)
        - **Historical trends**
        
        ### Features:
        - 🔮 **Yield Prediction**: Get instant predictions for any crop and district
        - 📈 **Analytics Dashboard**: Explore historical trends and patterns
        - 🗺️ **Interactive Maps**: Visualize yield distributions across districts
        - 📊 **Model Insights**: View feature importance and model performance
        """)
        
        if df_yield is not None:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df_yield))
            with col2:
                st.metric("Unique Crops", df_yield['crop'].nunique())
            with col3:
                st.metric("Years Covered", f"{df_yield['year'].min()}-{df_yield['year'].max()}")
            with col4:
                st.metric("Avg Yield", f"{df_yield['yield'].mean():.0f}")
    
    elif page == "🔮 Predict Yield":
        st.header("🔮 Crop Yield Prediction")
        
        if not models:
            st.warning("⚠️ Models not found. Please train models first by running `python src/train_models.py`")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Get available crops from data
                if df_yield is not None:
                    available_crops = sorted(df_yield['crop'].unique())
                else:
                    available_crops = ['Rice', 'Tea', 'Coconut', 'Rubber', 'Spices']
                
                crop = st.selectbox("Select Crop", available_crops)
                district = st.selectbox("Select District", get_sri_lanka_districts())
                year = st.slider("Year", 2020, 2030, 2024)
            
            with col2:
                rainfall = st.slider("Rainfall (mm)", 500, 3000, 1500)
                temperature = st.slider("Temperature (°C)", 20, 35, 27)
                humidity = st.slider("Humidity (%)", 50, 95, 75)
            
            if st.button("🔮 Predict Yield", type="primary"):
                with st.spinner("Calculating prediction..."):
                    # Use the predict utility
                    prediction = predict_yield(
                        crop=crop,
                        district=district,
                        year=year,
                        rainfall=rainfall,
                        temperature=temperature,
                        humidity=humidity
                    )
                    
                    if prediction is not None:
                        # Display results
                        st.success(f"✅ Predicted Yield: **{prediction:.2f}** units")
                        
                        # Show predictions from all models
                        st.subheader("Model Comparison")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Get predictions from all models
                        model_data = load_model_and_encoders('model/best_model.pkl')
                        if model_data:
                            from predict import prepare_prediction_features
                            features = prepare_prediction_features(
                                crop, district, year, rainfall, temperature,
                                humidity, model_data['encoders'], 
                                model_data['scaler'], model_data['feature_names']
                            )
                            
                            if features is not None:
                                if 'random_forest' in models:
                                    rf_pred = models['random_forest'].predict(features)[0]
                                    col1.metric("Random Forest", f"{rf_pred:.2f}")
                                
                                if 'xgboost' in models:
                                    xgb_pred = models['xgboost'].predict(features)[0]
                                    col2.metric("XGBoost", f"{xgb_pred:.2f}")
                                
                                if 'lightgbm' in models:
                                    lgb_pred = models['lightgbm'].predict(features)[0]
                                    col3.metric("LightGBM", f"{lgb_pred:.2f}")
                                
                                col4.metric("Best Model", f"{prediction:.2f}")
                    else:
                        st.error("❌ Prediction failed. Please ensure models are trained.")
    
    elif page == "📈 Analytics":
        st.header("📈 Analytics Dashboard")
        
        if df_yield is None:
            st.warning("⚠️ Data not available for analytics")
        else:
            # Yearly trends
            st.subheader("Yield Trends Over Years")
            yearly_avg = df_yield.groupby('year')['yield'].mean().reset_index()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(yearly_avg['year'], yearly_avg['yield'], marker='o', linewidth=2, markersize=8)
            ax.set_title('Average Crop Yield Trend Over Years', fontsize=14, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Average Yield')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Top crops
            st.subheader("Top Crops by Average Yield")
            top_crops = df_yield.groupby('crop')['yield'].mean().sort_values(ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            top_crops.plot(kind='barh', ax=ax)
            ax.set_title('Top 15 Crops by Average Yield', fontsize=14, fontweight='bold')
            ax.set_xlabel('Average Yield')
            ax.invert_yaxis()
            st.pyplot(fig)
            
            # Feature importance (if available)
            if os.path.exists('model/best_model.pkl'):
                st.subheader("Feature Importance")
                try:
                    model = models.get('best')
                    if hasattr(model, 'feature_importances_') and feature_names:
                        importance_df = pd.DataFrame({
                            'feature': feature_names[:len(model.feature_importances_)],
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False).head(15)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        importance_df.plot(x='feature', y='importance', kind='barh', ax=ax)
                        ax.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Importance')
                        ax.invert_yaxis()
                        st.pyplot(fig)
                except:
                    pass
    
    elif page == "🗺️ Map View":
        st.header("🗺️ District Yield Map")
        
        if df_yield is None:
            st.warning("⚠️ Data not available for map visualization")
        else:
            st.markdown("Interactive map showing average yields by district")
            yield_map = create_yield_map(df_yield)
            folium_static(yield_map)
            
            # District statistics
            st.subheader("District Statistics")
            if 'district' in df_yield.columns:
                district_stats = df_yield.groupby('district')['yield'].agg(['mean', 'std', 'count']).round(2)
                district_stats.columns = ['Mean Yield', 'Std Dev', 'Count']
                st.dataframe(district_stats.sort_values('Mean Yield', ascending=False))


if __name__ == "__main__":
    main()

