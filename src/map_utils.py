import folium
import pandas as pd
import numpy as np
from folium.plugins import HeatMap, MarkerCluster


# Sri Lanka district coordinates (approximate centers)
DISTRICT_COORDINATES = {
    'Colombo': [6.9271, 79.8612],
    'Gampaha': [7.0913, 79.9994],
    'Kalutara': [6.5854, 79.9607],
    'Kandy': [7.2906, 80.6337],
    'Matale': [7.4675, 80.6234],
    'Nuwara Eliya': [6.9497, 80.7891],
    'Galle': [6.0329, 80.2170],
    'Matara': [5.9549, 80.5550],
    'Hambantota': [6.1244, 81.1186],
    'Jaffna': [9.6615, 80.0255],
    'Kilinochchi': [9.4004, 80.3991],
    'Mannar': [8.9776, 79.9117],
    'Vavuniya': [8.7514, 80.4971],
    'Mullaitivu': [9.2671, 80.8131],
    'Batticaloa': [7.7172, 81.7004],
    'Ampara': [7.2975, 81.6820],
    'Trincomalee': [8.5874, 81.2152],
    'Kurunegala': [7.4863, 80.3623],
    'Puttalam': [8.0362, 79.8283],
    'Anuradhapura': [8.3114, 80.4037],
    'Polonnaruwa': [7.9403, 81.0000],
    'Badulla': [6.9934, 81.0550],
    'Moneragala': [6.8728, 81.3481],
    'Ratnapura': [6.6828, 80.4012],
    'Kegalle': [7.2513, 80.3464]
}


def create_yield_map(df_yield, district_col='district', yield_col='yield'):
    """
    Create an interactive map showing average yields by district
    
    Args:
        df_yield: DataFrame with district and yield columns
        district_col: Name of district column
        yield_col: Name of yield column
        
    Returns:
        folium.Map object
    """
    # Sri Lanka center coordinates
    sri_lanka_center = [7.8731, 80.7718]
    
    # Create base map
    m = folium.Map(
        location=sri_lanka_center,
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Calculate average yield by district
    if df_yield is not None and district_col in df_yield.columns:
        district_yields = df_yield.groupby(district_col)[yield_col].agg(['mean', 'count']).reset_index()
        district_yields.columns = [district_col, 'avg_yield', 'count']
        
        # Normalize yields for color mapping
        min_yield = district_yields['avg_yield'].min()
        max_yield = district_yields['avg_yield'].max()
        yield_range = max_yield - min_yield if max_yield > min_yield else 1
        
        # Add markers for each district
        for _, row in district_yields.iterrows():
            district = row[district_col]
            avg_yield = row['avg_yield']
            count = row['count']
            
            # Get coordinates
            if district in DISTRICT_COORDINATES:
                lat, lon = DISTRICT_COORDINATES[district]
            else:
                # Use center if district not found
                lat, lon = sri_lanka_center
            
            # Determine color based on yield
            normalized_yield = (avg_yield - min_yield) / yield_range
            if normalized_yield > 0.7:
                color = 'green'
            elif normalized_yield > 0.4:
                color = 'orange'
            else:
                color = 'red'
            
            # Create popup text
            popup_text = f"""
            <b>{district}</b><br>
            Average Yield: {avg_yield:,.0f}<br>
            Data Points: {count}
            """
            
            # Add circle marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + normalized_yield * 15,  # Size based on yield
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"{district}: {avg_yield:,.0f}",
                color='black',
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    
    return m


def create_heatmap(df_yield, district_col='district', yield_col='yield'):
    """
    Create a heatmap of yields across districts
    
    Args:
        df_yield: DataFrame with district and yield columns
        district_col: Name of district column
        yield_col: Name of yield column
        
    Returns:
        folium.Map object with heatmap layer
    """
    # Sri Lanka center coordinates
    sri_lanka_center = [7.8731, 80.7718]
    
    # Create base map
    m = folium.Map(
        location=sri_lanka_center,
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Prepare heatmap data
    if df_yield is not None and district_col in df_yield.columns:
        district_yields = df_yield.groupby(district_col)[yield_col].mean().reset_index()
        
        heat_data = []
        for _, row in district_yields.iterrows():
            district = row[district_col]
            avg_yield = row[yield_col]
            
            if district in DISTRICT_COORDINATES:
                lat, lon = DISTRICT_COORDINATES[district]
                # Weight by yield value
                heat_data.append([lat, lon, avg_yield])
        
        # Add heatmap layer
        if heat_data:
            HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    return m


def create_prediction_map(predictions_dict):
    """
    Create a map showing predicted yields for districts
    
    Args:
        predictions_dict: Dictionary with district names as keys and predicted yields as values
        
    Returns:
        folium.Map object
    """
    # Sri Lanka center coordinates
    sri_lanka_center = [7.8731, 80.7718]
    
    # Create base map
    m = folium.Map(
        location=sri_lanka_center,
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    if predictions_dict:
        # Normalize predictions for color mapping
        values = list(predictions_dict.values())
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val > min_val else 1
        
        # Add markers for each prediction
        for district, prediction in predictions_dict.items():
            if district in DISTRICT_COORDINATES:
                lat, lon = DISTRICT_COORDINATES[district]
            else:
                lat, lon = sri_lanka_center
            
            # Determine color
            normalized = (prediction - min_val) / val_range
            if normalized > 0.7:
                color = 'green'
            elif normalized > 0.4:
                color = 'orange'
            else:
                color = 'red'
            
            # Create popup
            popup_text = f"""
            <b>{district}</b><br>
            Predicted Yield: {prediction:,.0f}
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=10 + normalized * 15,
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"{district}: {prediction:,.0f}",
                color='black',
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    
    return m


if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'district': ['Colombo', 'Kandy', 'Galle'],
        'yield': [50000, 45000, 40000]
    })
    
    map_obj = create_yield_map(sample_data)
    map_obj.save('test_map.html')
    print("Map saved to test_map.html")

