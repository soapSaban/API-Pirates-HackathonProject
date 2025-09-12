import streamlit as st
import pandas as pd
import numpy as np
import ee
import geemap
import requests
import joblib
from streamlit_folium import st_folium
import folium
import time
import os
import json
import pydeck as pdk
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import google.auth
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
import rasterio
from io import BytesIO
from PIL import Image
import base64
import cv2
from datetime import datetime, timedelta
import math

# --- Page Configuration ---
st.set_page_config(
    page_title="WildFireGuard: Advanced Forest Fire Prediction & Simulation",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to encode local image to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
# Add the background image with CSS that covers everything
image_base64 = get_base64_of_image("pexels-philkallahar-983200.jpg")
st.markdown(
    f"""
    <style>
    /* Target the root HTML element and body */
    html, body, #root, .stApp {{
        background-image: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        margin: 0;
        padding: 0;
        min-height: 100vh;
        background-color: #000000;
    }}
    
    /* Make sure all Streamlit elements have transparent backgrounds */
    .stApp > header {{
        background-color: transparent;
        position: relative;
        z-index: 999;
    }}
    
    .stApp > MainMenu {{
        background-color: transparent;
    }}
    
    /* Ensure the main content area doesn't cut off the background */
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 1rem 1rem 1rem;
        max-width: 95%;
    }}
    
    /* Fix for Streamlit's default padding that might cut off the background */
    .stApp {{
        padding: 0;
        margin: 0;
    }}
    
    /* Ensure all containers have transparent backgrounds */
    .stApp > div {{
        background-color: transparent;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Apply custom CSS for styling with static background image
st.markdown(f"""
<style>
    /* Static background image covering entire screen without being cut off */
    body, #root, .stApp {{
        background: url('data:image/png;base64,{image_base64 if image_base64 else ""}') no-repeat center center fixed !important;
        background-size: cover !important;
        background-position: center center !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
    }}
    
    /* Adjust background position to account for Streamlit header */
    .stApp::before {{
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/png;base64,{image_base64 if image_base64 else ""}') no-repeat center center;
        background-size: cover;
        z-index: -1;
        margin-top: -50px; /* Adjust this value to shift the background image up */
    }}
    
    /* Fix for any container that might be clipping the background */
    .stApp > div {{
        overflow: visible !important;
    }}
    
    /* Main content container with transparency */
    .main .block-container {{
        background-color: rgba(255, 255, 255, 0.92);
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 1rem 1rem 1rem;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 95%;
    }}

    /* Title container with proper centering */
    .title-container {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    
    .main-header {{
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin: 0 auto 0.5rem auto;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        width: 100%;
        display: block;
    }}
    
    .subtitle {{
        text-align: center;
        font-size: 1.1rem;
        color: #ffffff; /* Changed to white */
        margin: 0 auto 1.5rem auto;
        width: 100%;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8); /* Added text shadow for better visibility */
        font-weight: 500;
    }}
    
    .sub-header {{
        font-size: 1.4rem;
        color: #ffffff; /* Changed to white */
        border-bottom: 2px solid #ff4b4b;
        padding-bottom: 0.3rem;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8); /* Added text shadow for better visibility */
        font-weight: 600;
    }}
    
    .section-header {{
        font-size: 1.2rem;
        color: #2c3e50;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }}
    
    .info-box {{
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff4b4b;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        color: #2c3e50;
    }}
    
    .metric-card {{
        background-color: white;
        padding: 0.8rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 0.8rem;
        height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    
    .metric-value {{
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff4b4b;
        line-height: 1.2;
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: #7f8c8d;
        margin-bottom: 0.3rem;
    }}
    
    .risk-high {{
        color: #ff4b4b;
        font-weight: bold;
        background-color: #ffebee;
        padding: 0.5rem;
        border-radius: 0.3rem;
        text-align: center;
        margin: 0.5rem 0;
    }}
    
    .risk-medium {{
        color: #f39c12;
        font-weight: bold;
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        text-align: center;
        margin: 0.5rem 0;
    }}
    
    .risk-low {{
        color: #27ae60;
        font-weight: bold;
        background-color: #e8f5e9;
        padding: 0.5rem;
        border-radius: 0.3rem;
        text-align: center;
        margin: 0.5rem 0;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 40px;
        white-space: pre-wrap;
        background-color: #1E90FF;
        border-radius: 4px 4px 0px 0px;
        padding: 8px 12px;
        color: white;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: #0066CC;
        color: white;
    }}
    
    .stats-table {{
        width: 100%;
        border-collapse: collapse;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }}
    
    .stats-table th, .stats-table td {{
        border: 1px solid #ddd;
        padding: 6px;
        text-align: left;
    }}
    
    .stats-table th {{
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }}
    
    .stats-table tr:nth-child(even) {{
        background-color: #f8f9fa;
    }}
    
    .simulation-control {{
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
    }}
    
    .simulation-control h4 {{
        margin: 0 0 0.5rem 0;
        color: #2c3e50;
        font-size: 1rem;
    }}
    
    .stImage > div > div > img {{
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}
    
    .stButton > button {{
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-weight: bold;
        transition: background-color 0.3s;
    }}
    
    .stButton > button:hover {{
        background-color: #e63c3c;
        color: white;
    }}
    
    .element-container:has(> .stDeckGlJsonChart) {{
        border-radius: 0.5rem;
        overflow: hidden;
    }}
    
    [data-testid="column"] {{
        padding: 0 0.5rem;
    }}
    
    .stTabs [data-baseweb="tab-panel"] {{
        padding: 1rem 0;
    }}
    
    .no-data-warning {{
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
        text-align: center;
        margin: 1rem 0;
    }}
    
    .satellite-error {{
        background-color: #f8d7da;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        text-align: center;
        margin: 1rem 0;
    }}
    
    /* Additional styles for the manual coordinate section */
    .stNumberInput > label {{
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
        font-weight: 500;
    }}
    
    /* Style for the map section header */
    .element-container:has(> [data-testid="stMarkdownContainer"] > div > div > .sub-header) {{
        color: #ffffff;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
    }}
</style>
""", unsafe_allow_html=True)

# --- Constants & Secrets ---
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
GEE_PROJECT_ID = st.secrets.get("GEE_PROJECT_ID", "")

# --- Enhanced GEE Authentication ---
@st.cache_resource
def initialize_gee():
    """
    Initializes GEE using a simplified approach with service account or manual authentication.
    """
    try:
        # Try to initialize with existing credentials
        ee.Initialize(project=GEE_PROJECT_ID)
        return True
    except Exception as e:
        st.warning("Google Earth Engine is not properly configured. Satellite data features will be limited.")
        return False

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model and scaler."""
    try:
        model = joblib.load('model/best_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        feature_names = joblib.load('model/feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please make sure you've trained and saved the model first.")
        return None, None, None

# --- Data Fetching Functions ---
def get_live_weather_data(lat, lon):
    """Fetches live weather data from OpenWeatherMap."""
    try:
        if not OPENWEATHER_API_KEY:
            st.warning("OpenWeatherMap API key not configured. Using default weather values.")
            return {
                'temp': 25.0,
                'humidity': 45.0,
                'wind_speed': 5.0,
                'wind_deg': 90.0,
                'pressure': 1013.0,
                'clouds': 30.0
            }
            
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind'].get('deg', 0),
            'pressure': data['main']['pressure'],
            'clouds': data['clouds']['all']
        }
    except Exception as e:
        st.warning("Could not fetch live weather data. Using default values.")
        return {
            'temp': 25.0,
            'humidity': 45.0,
            'wind_speed': 5.0,
            'wind_deg': 90.0,
            'pressure': 1013.0,
            'clouds': 30.0
        }

def get_precipitation_data(lat, lon):
    """Get precipitation data from Open-Meteo API (Free)"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'precipitation,rain,showers,snowfall',
            'hourly': 'precipitation,rain,showers,snowfall,relative_humidity_2m,temperature_2m',
            'daily': 'precipitation_sum,rain_sum,showers_sum,snowfall_sum',
            'timezone': 'auto',
            'forecast_days': 3
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Current precipitation in mm
        current_rain = data.get('current', {}).get('precipitation', 0)
        current_snow = data.get('current', {}).get('snowfall', 0)
        
        # Get hourly precipitation for the last 24 hours and next 24 hours
        hourly_rain = data.get('hourly', {}).get('precipitation', [0]*72)
        hourly_snow = data.get('hourly', {}).get('snowfall', [0]*72)
        hourly_temp = data.get('hourly', {}).get('temperature_2m', [20]*72)
        hourly_humidity = data.get('hourly', {}).get('relative_humidity_2m', [50]*72)
        
        # Calculate average rainfall for the last 24 hours (more relevant for fire risk)
        past_24h_rain = hourly_rain[:24]  # First 24 hours are past data
        avg_24h_rain = sum(past_24h_rain) / len(past_24h_rain) if past_24h_rain else 0
        
        # Calculate average temperature and humidity
        avg_temp = sum(hourly_temp[:24]) / len(hourly_temp[:24]) if hourly_temp[:24] else 20
        avg_humidity = sum(hourly_humidity[:24]) / len(hourly_humidity[:24]) if hourly_humidity[:24] else 50
        
        return {
            'current_rain_mm': max(0, current_rain),
            'current_snow_mm': max(0, current_snow),
            'avg_24h_rain_mm': max(0, avg_24h_rain),
            'avg_temp': avg_temp,
            'avg_humidity': avg_humidity,
            'hourly_forecast': [max(0, r) for r in hourly_rain[24:48]],  # Next 24 hours
            'hourly_snow_forecast': [max(0, s) for s in hourly_snow[24:48]],  # Next 24 hours
            'hourly_temp_forecast': hourly_temp[24:48],
            'hourly_humidity_forecast': hourly_humidity[24:48]
        }
    except Exception as e:
        st.warning("Could not fetch precipitation data. Using conservative estimates.")
        # Return conservative estimates if API fails
        return {'current_rain_mm': 0, 'current_snow_mm': 0, 'avg_24h_rain_mm': 0, 
                'avg_temp': 20, 'avg_humidity': 50, 'hourly_forecast': [0]*24,
                'hourly_snow_forecast': [0]*24, 'hourly_temp_forecast': [20]*24,
                'hourly_humidity_forecast': [50]*24}

def get_rainfall_data(lat, lon):
    """Get rainfall data for model prediction"""
    try:
        precipitation_data = get_precipitation_data(lat, lon)
        # Use average of last 24 hours rainfall for better prediction
        return max(0, precipitation_data['avg_24h_rain_mm'])
    except:
        return 0.0

def format_feature_names(features_dict):
    """
    Converts raw feature names and values to human-readable format with proper units.
    """
    if not features_dict:
        return None
    
    # Mapping of feature names to human-readable names and units
    feature_mapping = {
        'X': {'name': 'Spatial Coordinate X', 'unit': '', 'format': '{:.1f}'},
        'Y': {'name': 'Spatial Coordinate Y', 'unit': '', 'format': '{:.1f}'},
        'FFMC': {'name': 'Fine Fuel Moisture Code', 'unit': '', 'format': '{:.1f}'},
        'DMC': {'name': 'Duff Moisture Code', 'unit': '', 'format': '{:.1f}'},
        'DC': {'name': 'Drought Code', 'unit': '', 'format': '{:.1f}'},
        'ISI': {'name': 'Initial Spread Index', 'unit': '', 'format': '{:.1f}'},
        'temp': {'name': 'Temperature', 'unit': '°C', 'format': '{:.1f}'},
        'RH': {'name': 'Relative Humidity', 'unit': '%', 'format': '{:.0f}'},
        'wind': {'name': 'Wind Speed', 'unit': 'km/h', 'format': '{:.1f}'},
        'rain': {'name': 'Rainfall (24h avg)', 'unit': 'mm', 'format': '{:.2f}'}
    }
    
    formatted_features = {}
    
    for key, value in features_dict.items():
        if key in feature_mapping and value is not None:
            mapping = feature_mapping[key]
            try:
                formatted_value = mapping['format'].format(value)
                formatted_name = f"{mapping['name']}"
                if mapping['unit']:
                    formatted_name += f" ({mapping['unit']})"
                formatted_features[formatted_name] = formatted_value
            except (ValueError, TypeError):
                formatted_name = f"{mapping['name']}"
                if mapping['unit']:
                    formatted_name += f" ({mapping['unit']})"
                formatted_features[formatted_name] = str(value)
        else:
            formatted_features[key] = value
    
    return formatted_features

def adapt_features_to_model(weather_data, gee_data, lat, lon):
    """
    Convert available weather and GEE data to the format expected by the trained model.
    This function maps your available data to the model's expected features with real rainfall data.
    """
    # Get real rainfall data from Open-Meteo API
    rain_data = get_rainfall_data(lat, lon)
    
    # Default values based on the forestfires.csv dataset statistics
    default_values = {
        'X': max(1, min(9, round(abs(lat) * 0.5))),  # Map lat to X (1-9)
        'Y': max(2, min(9, round(abs(lon) * 0.2))),  # Map lon to Y (2-9)
        'FFMC': 90.0, 'DMC': 100.0, 'DC': 500.0, 'ISI': 8.0,
        'temp': 20.0, 'RH': 50.0, 'wind': 4.0, 'rain': rain_data
    }
    
    # Start with default values
    adapted_features = default_values.copy()
    
    # Map available weather data to model features
    if weather_data:
        adapted_features['temp'] = weather_data.get('temp', adapted_features['temp'])
        adapted_features['RH'] = weather_data.get('humidity', adapted_features['RH'])
        adapted_features['wind'] = weather_data.get('wind_speed', adapted_features['wind'])
        # Keep the real rain data from Open-Meteo
    
    # Map GEE data to model features (scientifically informed approximations)
    if gee_data:
        ndvi = gee_data.get('ndvi', 0.5)
        lst = gee_data.get('lst', 25.0)
        slope = gee_data.get('slope', 10.0)
        water_mask = gee_data.get('water_mask', 0.0)
        urban_mask = gee_data.get('urban_mask', 0.0)
        snow_mask = gee_data.get('snow_mask', 0.0)
        
        # Refined approximations based on fire science principles
        # FFMC: Higher temperature → lower moisture → higher FFMC
        adapted_features['FFMC'] = 82.0 + (lst - 20.0) * 0.6
        
        # DMC: Lower vegetation → higher duff moisture code
        adapted_features['DMC'] = 50.0 + (1.0 - ndvi) * 120.0
        
        # DC: Long-term drought effect, influenced by vegetation
        adapted_features['DC'] = 300.0 + (1.0 - ndvi) * 500.0
        
        # ISI: Wind speed is primary factor, temperature has secondary effect
        adapted_features['ISI'] = 4.0 + adapted_features['wind'] * 0.8 + (lst - 20.0) * 0.1
        
        # Adjust for water, urban, and snow areas (reduce fire risk)
        if water_mask > 0.5 or urban_mask > 0.5 or snow_mask > 0.5:
            adapted_features['FFMC'] *= 0.3
            adapted_features['DMC'] *= 0.3
            adapted_features['DC'] *= 0.3
            adapted_features['ISI'] *= 0.1
    
    # Ensure values stay within reasonable bounds
    adapted_features['FFMC'] = max(0, min(100, adapted_features['FFMC']))
    adapted_features['DMC'] = max(0, min(300, adapted_features['DMC']))
    adapted_features['DC'] = max(0, min(1000, adapted_features['DC']))
    adapted_features['ISI'] = max(0, min(50, adapted_features['ISI']))
    adapted_features['rain'] = max(0, min(50, adapted_features['rain']))  # Cap rain at 50mm
    
    return adapted_features

def get_gee_data(lat, lon):
    """Fetches comprehensive landscape data from Google Earth Engine for a given point."""
    try:
        # Check if GEE is initialized
        if not ee.data._initialized:
            return None
            
        point = ee.Geometry.Point(lon, lat)
        
        # Get current date and date 30 days ago
        current_date = ee.Date(time.time() * 1000)
        start_date = current_date.advance(-30, 'day')
        
        # Sentinel-2 for NDVI and land cover
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(point) \
            .filterDate(start_date, current_date) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        
        # Check if collection is not empty
        collection_size = s2_collection.size().getInfo()
        if collection_size == 0:
            return None
            
        s2_image = ee.Image(s2_collection.first())
        ndvi = s2_image.normalizedDifference(['B8', 'B4']).rename('ndvi')
        
        # Land surface temperature
        lst_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterBounds(point) \
            .filterDate(start_date, current_date)
            
        # Check if MODIS collection is not empty
        lst_collection_size = lst_collection.size().getInfo()
        if lst_collection_size == 0:
            lst = ee.Image.constant(25).rename('lst')
        else:
            lst_image = lst_collection.first()
            lst = lst_image.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('lst')

        # Digital elevation model
        dem = ee.Image('USGS/SRTMGL1_003')
        slope = ee.Terrain.slope(dem).rename('slope')
        aspect = ee.Terrain.aspect(dem).rename('aspect')
        
        # Water detection using NDWI
        ndwi = s2_image.normalizedDifference(['B3', 'B8']).rename('ndwi')
        water_mask = ndwi.gt(0.2).rename('water_mask')
        
        # Snow detection using NDSI
        ndsi = s2_image.normalizedDifference(['B3', 'B11']).rename('ndsi')
        snow_mask = ndsi.gt(0.4).rename('snow_mask')
        
        # Urban area detection (using built-up index approximation)
        # Built-up index: (SWIR1 - NIR) / (SWIR1 + NIR)
        built_up_index = s2_image.expression(
            '(B11 - B8) / (B11 + B8)', {
                'B11': s2_image.select('B11'),
                'B8': s2_image.select('B8')
            }).rename('built_up_index')
        urban_mask = built_up_index.gt(0.1).rename('urban_mask')
        
        # Fuel load estimation based on NDVI
        fuel_load = ndvi.multiply(2.0).rename('fuel_load')

        # Stack all features
        feature_stack = ee.Image.cat([ndvi, lst, slope, aspect, water_mask, 
                                    snow_mask, urban_mask, fuel_load, ndwi, ndsi])
        
        # Get values for the point
        feature_values = feature_stack.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30  # Higher resolution
        ).getInfo()
        
        # Ensure all expected features are present
        expected_features = ['ndvi', 'lst', 'slope', 'aspect', 'water_mask', 
                           'snow_mask', 'urban_mask', 'fuel_load', 'ndwi', 'ndsi']
        for feature in expected_features:
            if feature not in feature_values or feature_values[feature] is None:
                feature_values[feature] = 0
                
        return feature_values
    except Exception as e:
        return None

def get_satellite_image(lat, lon, size=100, scale=30):
    """Fetches a satellite image for the selected location."""
    try:
        if not ee.data._initialized:
            return None
            
        point = ee.Geometry.Point(lon, lat)
        
        # Calculate bounds for the image
        offset = (size * scale) / 2 / 111320  # Convert meters to degrees (approx)
        region = ee.Geometry.Rectangle([
            lon - offset, lat - offset,
            lon + offset, lat + offset
        ])
        
        # Get recent Sentinel-2 image
        current_date = ee.Date(time.time() * 1000)
        start_date = current_date.advance(-30, 'day')
        
        s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(point) \
            .filterDate(start_date, current_date) \
            .sort('CLOUDY_PIXEL_PERCENTAGE') \
            .first()
        
        # Create true color image
        image = s2_collection.select(['B4', 'B3', 'B2']).visualize(min=0, max=3000)
        
        # Get URL for the image
        url = image.getThumbURL({
            'region': region,
            'dimensions': [size, size],
            'format': 'png'
        })
        
        # Download the image
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return np.array(img) / 255.0
        
    except Exception as e:
        return None

# --- Enhanced Fire Simulation Class ---
class EnhancedFireSimulator:
    def __init__(self, lat, lon, size=100, scale=30):
        self.center_lat = lat
        self.center_lon = lon
        self.size = size
        self.scale = scale
        self.grid = np.zeros((size, size), dtype=float)
        
        # Get satellite image for the location
        self.satellite_image = get_satellite_image(lat, lon, size, scale)
        
        # Get comprehensive landscape data
        self.landscape_data = self._get_landscape_data()
        
        # Check if we have satellite data
        self.has_satellite_data = self.satellite_image is not None
        self.has_landscape_data = self.landscape_data is not None
        
        if self.has_satellite_data and self.has_landscape_data:
            # Start with multiple ignition points for more realistic simulation
            center = size // 2
            self.grid[center, center] = 1.0  # Main ignition point
            if size > 20:
                self.grid[center+2, center] = 0.8  # Secondary ignition
                self.grid[center, center+2] = 0.8  # Secondary ignition
            
            # Create fuel map based on NDVI
            self.fuel_map = self._create_fuel_map()
            
            # Create custom colormap for visualization
            self.cmap = self._create_fire_colormap()
        else:
            # Create empty fuel map and colormap
            self.fuel_map = np.zeros((size, size))
            self.cmap = self._create_fire_colormap()
        
        # State tracking
        self.burned_cells = 0
        self.burning_cells = np.sum(self.grid >= 0.8) if (self.has_satellite_data and self.has_landscape_data) else 0
        self.time_elapsed = 0
        self.history = []
        self.fire_perimeter = set()
        self.fire_front = set()
        self._update_fire_front()

    def _get_landscape_data(self):
        """Get comprehensive landscape data from GEE"""
        try:
            # Get data from GEE
            gee_data = get_gee_data(self.center_lat, self.center_lon)
            if gee_data is None:
                return None
                
            # Create grid of values based on the point data with some spatial variation
            x, y = np.meshgrid(np.linspace(-1, 1, self.size), np.linspace(-1, 1, self.size))
            
            # Base values from GEE
            base_ndvi = gee_data.get('ndvi', 0.5)
            base_lst = gee_data.get('lst', 25.0)
            base_slope = gee_data.get('slope', 10.0)
            base_water = gee_data.get('water_mask', 0.0)
            base_snow = gee_data.get('snow_mask', 0.0)
            base_urban = gee_data.get('urban_mask', 0.0)
            base_fuel = gee_data.get('fuel_load', 1.0)
            
            # Create realistic spatial variations
            # NDVI variation (higher in valleys, lower on ridges)
            ndvi_variation = 0.2 * (np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) +
                                  0.5 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y))
            ndvi_grid = np.clip(base_ndvi + ndvi_variation, 0.1, 0.9)
            
            # Slope variation (create realistic topography)
            slope_variation = 5 * (np.sin(3 * np.pi * x) * np.cos(3 * np.pi * y) +
                                 0.7 * np.sin(5 * np.pi * x) * np.cos(5 * np.pi * y))
            slope_grid = np.clip(base_slope + slope_variation, 0, 45)  # 0-45 degrees
            
            # Temperature variation (cooler in valleys, warmer on ridges)
            lst_variation = 3 * (np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y))
            lst_grid = np.clip(base_lst + lst_variation, 15, 40)
            
            # Water bodies (rivers in valleys)
            water_grid = np.zeros((self.size, self.size))
            if base_water > 0.2:  # If water is detected at the point
                # Create river-like patterns following topography
                for i in range(3):
                    river_x = 0.3 * np.sin(2 * np.pi * y + i * np.pi/3)
                    river_mask = np.abs(x - river_x) < 0.1
                    water_grid[river_mask] = 1.0
            
            # Snow cover (higher elevations)
            snow_grid = np.zeros((self.size, self.size))
            if base_snow > 0.2:  # If snow is detected at the point
                # Snow at higher elevations (simulated by distance from center)
                elevation = np.sqrt(x**2 + y**2)
                snow_mask = elevation > 0.7
                snow_grid[snow_mask] = 1.0
            
            # Urban areas (flatter terrain)
            urban_grid = np.zeros((self.size, self.size))
            if base_urban > 0.2:  # If urban area is detected at the point
                # Urban areas in flatter regions
                urban_mask = (slope_grid < 5) & (np.abs(x) < 0.3) & (np.abs(y) < 0.3)
                urban_grid[urban_mask] = 1.0
            
            # Fuel load based on NDVI but reduced in water, snow, and urban areas
            fuel_grid = base_fuel * ndvi_grid * (1 - water_grid) * (1 - snow_grid) * (1 - urban_grid)
            
            return {
                'ndvi': ndvi_grid,
                'lst': lst_grid,
                'slope': slope_grid,
                'water_mask': water_grid,
                'snow_mask': snow_grid,
                'urban_mask': urban_grid,
                'fuel_load': fuel_grid,
                'aspect': np.arctan2(y, x)  # Simplified aspect
            }
        
        except Exception as e:
            return None

    def _create_fuel_map(self):
        """Create a fuel map based on NDVI values and landscape features"""
        if not (self.has_satellite_data and self.has_landscape_data):
            return np.zeros((self.size, self.size))
            
        # Base fuel from NDVI (higher NDVI = more fuel)
        fuel_map = self.landscape_data['fuel_load']
        
        # Reduce fuel in water, snow, and urban areas
        fuel_map *= (1 - self.landscape_data['water_mask'])
        fuel_map *= (1 - self.landscape_data['snow_mask'])
        fuel_map *= (1 - self.landscape_data['urban_mask'])
        
        # Adjust fuel based on slope (less fuel on very steep slopes)
        fuel_map *= np.clip(1 - 0.02 * self.landscape_data['slope'], 0.5, 1.0)
        
        return fuel_map

    def _create_fire_colormap(self):
        """Create a custom colormap for fire visualization"""
        colors = [
            (0.0, (0.1, 0.5, 0.1)),     # Healthy vegetation (dark green)
            (0.2, (0.8, 0.9, 0.3)),     # Dry vegetation (light green-yellow)
            (0.4, (1.0, 0.8, 0.2)),     # Ignition (yellow)
            (0.6, (1.0, 0.5, 0.0)),     # Burning (orange)
            (0.8, (1.0, 0.2, 0.0)),     # Intense fire (red-orange)
            (1.0, (0.5, 0.2, 0.1))      # Burned (dark brown)
        ]
        return LinearSegmentedColormap.from_list('fire_cmap', colors)

    def _update_fire_front(self):
        """Update the fire front and perimeter cells"""
        if not (self.has_satellite_data and self.has_landscape_data):
            return
            
        # Find all burning cells
        burning_cells = set(zip(*np.where(self.grid >= 0.8)))
        
        # Fire perimeter includes all burning cells
        self.fire_perimeter = burning_cells
        
        # Fire front includes burning cells adjacent to unburned cells
        self.fire_front = set()
        for r, c in burning_cells:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.size and 0 <= nc < self.size and 
                        self.grid[nr, nc] < 0.8):
                        self.fire_front.add((r, c))
                        break

    def _calculate_spread_probability(self, r, c, weather):
        """Calculate probability of fire spreading to cell (r, c) with extreme realism"""
        if not (self.has_satellite_data and self.has_landscape_data):
            return 0  # No spread without data
            
        # Base probability based on fuel
        base_prob = 0.1 * (self.fuel_map[r, c] / np.max(self.fuel_map))
        
        # No spread in water, snow, or urban areas
        if (self.landscape_data['water_mask'][r, c] > 0.5 or
            self.landscape_data['snow_mask'][r, c] > 0.5 or
            self.landscape_data['urban_mask'][r, c] > 0.5):
            return 0.0
        
        # Slope effect (fire spreads faster uphill)
        slope = self.landscape_data['slope'][r, c]
        slope_effect = 0.0
        
        if r > 0 and c > 0 and r < self.size-1 and c < self.size-1:
            # Calculate slope direction
            dz_dx = (self.landscape_data['slope'][r, c+1] - self.landscape_data['slope'][r, c-1]) / 2
            dz_dy = (self.landscape_data['slope'][r+1, c] - self.landscape_data['slope'][r-1, c]) / 2
            
            # Slope magnitude effect
            slope_magnitude_effect = 0.001 * slope
            
            # Slope direction effect (fire spreads faster uphill)
            if abs(dz_dx) > 0.1 or abs(dz_dy) > 0.1:
                # Calculate direction from fire source to target
                fire_direction = math.atan2(self.size//2 - r, self.size//2 - c)
                slope_direction = math.atan2(dz_dy, dz_dx)
                
                # Alignment between fire direction and slope direction
                direction_alignment = math.cos(fire_direction - slope_direction)
                slope_direction_effect = 0.002 * slope * max(0, direction_alignment)
                
                slope_effect = slope_magnitude_effect + slope_direction_effect
        
        # Wind effect (major factor)
        wind_speed = weather.get('wind_speed', 0)
        wind_dir = math.radians(weather.get('wind_deg', 0))
        
        # Calculate wind direction components
        wind_u = -wind_speed * math.sin(wind_dir)  # u component (east-west)
        wind_v = -wind_speed * math.cos(wind_dir)  # v component (north-south)
        
        # Calculate direction from fire center to target cell
        center_r, center_c = self.size // 2, self.size // 2
        fire_to_cell_dir = math.atan2(r - center_r, c - center_c)
        
        # Wind alignment with fire spread direction
        wind_alignment = math.cos(wind_dir - fire_to_cell_dir)
        wind_effect = 0.003 * wind_speed * max(0, wind_alignment)
        
        # Humidity effect (lower humidity = higher probability)
        humidity = weather.get('humidity', 50)
        humidity_effect = 0.004 * (100 - humidity)
        
        # Temperature effect (higher temperature = higher probability)
        temperature = weather.get('temp', 20)
        temp_effect = 0.002 * (temperature - 20)
        
        # Fuel moisture effect (based on recent rainfall)
        rainfall = weather.get('rain_24h', 0)
        fuel_moisture_effect = -0.005 * min(rainfall, 20)  # Cap at 20mm
        
        # Vegetation type effect (based on NDVI)
        ndvi = self.landscape_data['ndvi'][r, c]
        vegetation_effect = 0.002 * (ndvi * 100 - 50)  # Higher for greener vegetation
        
        # Time of day effect (fires spread faster during daytime)
        current_hour = datetime.now().hour
        time_of_day_effect = 0.001 * (12 - abs(12 - current_hour))  # Peak at noon
        
        # Combine all effects
        total_prob = (base_prob + slope_effect + wind_effect + humidity_effect +
                     temp_effect + fuel_moisture_effect + vegetation_effect +
                     time_of_day_effect)
        
        # Ensure probability is within reasonable bounds
        return np.clip(total_prob, 0.01, 0.9)

    def _calculate_spread_direction(self, source_r, source_c, target_r, target_c, weather):
        """Calculate directional bias for fire spread"""
        if not (self.has_satellite_data and self.has_landscape_data):
            return 0, 0  # No direction without data
            
        # Default direction vector (from source to target)
        dir_r, dir_c = target_r - source_r, target_c - source_c
        dir_magnitude = math.sqrt(dir_r**2 + dir_c**2)
        if dir_magnitude > 0:
            dir_r, dir_c = dir_r / dir_magnitude, dir_c / dir_magnitude
        else:
            return 0, 0  # No direction
        
        # Wind effect
        wind_speed = weather.get('wind_speed', 0)
        wind_dir = math.radians(weather.get('wind_deg', 0))
        wind_r = -math.sin(wind_dir)  # Convert wind direction to vector
        wind_c = -math.cos(wind_dir)
        
        # Slope effect (get slope at target cell)
        if (0 <= target_r < self.size and 0 <= target_c < self.size and
            target_r > 0 and target_c > 0 and 
            target_r < self.size-1 and target_c < self.size-1):
            
            dz_dx = (self.landscape_data['slope'][target_r, target_c+1] - 
                    self.landscape_data['slope'][target_r, target_c-1]) / 2
            dz_dy = (self.landscape_data['slope'][target_r+1, target_c] - 
                    self.landscape_data['slope'][target_r-1, target_c]) / 2
            
            slope_magnitude = math.sqrt(dz_dx**2 + dz_dy**2)
            if slope_magnitude > 0:
                slope_r = dz_dy / slope_magnitude  # Upslope direction
                slope_c = dz_dx / slope_magnitude
            else:
                slope_r, slope_c = 0, 0
        else:
            slope_r, slope_c = 0, 0
        
        # Combine effects (weight wind more than slope)
        combined_r = dir_r + 1.5 * wind_r + 0.8 * slope_r
        combined_c = dir_c + 1.5 * wind_c + 0.8 * slope_c
        
        # Normalize
        combined_magnitude = math.sqrt(combined_r**2 + combined_c**2)
        if combined_magnitude > 0:
            return combined_r / combined_magnitude, combined_c / combined_magnitude
        else:
            return dir_r, dir_c  # Fallback to original direction

    def step(self, weather, time_step=1):
        """Advance the simulation by one time step with extreme realism"""
        if not (self.has_satellite_data and self.has_landscape_data):
            # Return empty grid if no data is available
            return self.grid
            
        # Update current burning cells (increase intensity)
        burning_mask = self.grid >= 0.8
        self.grid[burning_mask] += 0.05 * time_step  # Slower intensity increase for realism
        
        # Mark cells that have burned out
        burned_out = self.grid > 1.5
        self.grid[burned_out] = 2.0  # Completely burned
        
        # Update weather with recent rainfall data
        precipitation_data = get_precipitation_data(self.center_lat, self.center_lon)
        weather['rain_24h'] = precipitation_data['avg_24h_rain_mm']
        
        # Find newly ignited cells
        new_fires = np.zeros_like(self.grid, dtype=bool)
        
        # Update fire front
        self._update_fire_front()
        
        # Spread from fire front cells only (more efficient and realistic)
        for r, c in self.fire_front:
            # Check all 8 neighboring cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = r + dr, c + dc
                    
                    # Check if within bounds and not already burning/burned
                    if (0 <= nr < self.size and 0 <= nc < self.size and 
                        self.grid[nr, nc] < 0.8):
                        
                        # Calculate spread probability with extreme realism
                        spread_prob = self._calculate_spread_probability(nr, nc, weather)
                        
                        # Apply directional bias
                        dir_r, dir_c = self._calculate_spread_direction(r, c, nr, nc, weather)
                        direction_bias = max(0, dr * dir_r + dc * dir_c)
                        spread_prob *= (0.7 + 0.3 * direction_bias)
                        
                        # Apply probability with time step adjustment
                        if np.random.rand() < spread_prob * time_step:
                            new_fires[nr, nc] = True
        
        # Ignite new cells
        self.grid[new_fires] = 0.8
        
        # Update statistics
        self.burning_cells = np.sum((self.grid >= 0.8) & (self.grid < 1.5))
        self.burned_cells = np.sum(self.grid >= 1.5)
        self.time_elapsed += time_step
        
        # Record history
        self.history.append({
            'time': self.time_elapsed,
            'burning': self.burning_cells,
            'burned': self.burned_cells,
            'total_affected': self.burning_cells + self.burned_cells,
            'area_ha': (self.burning_cells + self.burned_cells) * (self.scale ** 2) / 10000
        })
        
        return self.grid

    def get_stats(self):
        """Get current simulation statistics"""
        total_area = self.size * self.size * (self.scale ** 2) / 10000  # Convert to hectares
        burning_area = self.burning_cells * (self.scale ** 2) / 10000
        burned_area = self.burned_cells * (self.scale ** 2) / 10000
        
        # Calculate fire line length (perimeter)
        perimeter_cells = 0
        for r, c in self.fire_perimeter:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (not (0 <= nr < self.size and 0 <= nc < self.size) or 
                    self.grid[nr, nc] < 0.8):
                    perimeter_cells += 1
        
        fire_line_length = perimeter_cells * self.scale  # meters
        
        # Calculate rate of spread
        ros = 0
        if len(self.history) > 1:
            current_area = self.history[-1]['area_ha']
            prev_area = self.history[-2]['area_ha'] if len(self.history) > 1 else 0
            time_diff = self.history[-1]['time'] - self.history[-2]['time'] if len(self.history) > 1 else 1
            ros = (current_area - prev_area) / time_diff  # ha/hour
        
        return {
            'time_elapsed': self.time_elapsed,
            'burning_cells': self.burning_cells,
            'burned_cells': self.burned_cells,
            'total_affected': self.burning_cells + self.burned_cells,
            'burning_area': burning_area,
            'burned_area': burned_area,
            'percent_burned': 100 * self.burned_cells / (self.size * self.size),
            'fire_line_length': fire_line_length,
            'rate_of_spread': ros,
            'has_satellite_data': self.has_satellite_data,
            'has_landscape_data': self.has_landscape_data
        }

    def visualize_on_map(self, ax=None):
        """Create a visualization of the current fire state overlaid on the satellite image"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        if not (self.has_satellite_data and self.has_landscape_data):
            # Display a professional message when no data is available
            ax.text(0.5, 0.5, "Satellite imagery not available for this location\n\n"
                              "Possible reasons:\n"
                              "• Google Earth Engine not configured\n"
                              "• Location outside satellite coverage\n"
                              "• Cloud cover obscuring the area\n"
                              "• Recent imagery not available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red',
                    bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
            ax.set_title("Satellite Data Unavailable")
            ax.axis('off')
            return ax
        
        # Display the satellite image as base
        ax.imshow(self.satellite_image)
        
        # Overlay fire on top with transparency
        fire_mask = self.grid > 0.7
        fire_intensity = np.clip(self.grid[fire_mask], 0.7, 2.0)
        
        # Create a colored overlay for fire
        overlay = np.zeros((self.size, self.size, 4))
        
        # Color mapping for fire with transparency
        for i, intensity in enumerate(fire_intensity):
            # Get coordinates
            coords = np.where(fire_mask)
            r, c = coords[0][i], coords[1][i]
            
            if intensity < 1.0:  # Early fire (yellow to orange)
                overlay[r, c] = [1.0, 0.8, 0.2, 0.7]  # RGBA: yellow with transparency
            elif intensity < 1.5:  # Medium fire (orange to red)
                overlay[r, c] = [1.0, 0.5, 0.0, 0.8]  # RGBA: orange with transparency
            else:  # Burned out (brown)
                overlay[r, c] = [0.5, 0.2, 0.1, 0.9]  # RGBA: brown with transparency
        
        # Overlay water bodies
        water_mask = self.landscape_data['water_mask'] > 0.5
        overlay[water_mask] = [0.2, 0.4, 0.8, 0.6]  # Blue for water
        
        # Overlay urban areas
        urban_mask = self.landscape_data['urban_mask'] > 0.5
        overlay[urban_mask] = [0.6, 0.6, 0.6, 0.5]  # Gray for urban
        
        # Overlay snow areas
        snow_mask = self.landscape_data['snow_mask'] > 0.5
        overlay[snow_mask] = [0.9, 0.9, 0.9, 0.6]  # White for snow
        
        # Apply the overlay
        ax.imshow(overlay)
        
        # Add a title with coordinates
        ax.set_title(f"Fire Simulation at ({self.center_lat:.4f}, {self.center_lon:.4f})\nTime: {self.time_elapsed} hours")
        ax.axis('off')
        
        # Add scale bar (approximate)
        scale_meters = self.size * self.scale
        if scale_meters > 1000:
            scale_text = f"Scale: {scale_meters/1000:.1f} km"
        else:
            scale_text = f"Scale: {scale_meters:.0f} m"
        ax.text(0.02, 0.02, scale_text, transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, fc=[1.0, 0.8, 0.2, 0.7], label='Ignition'),
            plt.Rectangle((0,0),1,1, fc=[1.0, 0.5, 0.0, 0.8], label='Burning'),
            plt.Rectangle((0,0),1,1, fc=[0.5, 0.2, 0.1, 0.9], label='Burned'),
            plt.Rectangle((0,0),1,1, fc=[0.2, 0.4, 0.8, 0.6], label='Water'),
            plt.Rectangle((0,0),1,1, fc=[0.6, 0.6, 0.6, 0.5], label='Urban'),
            plt.Rectangle((0,0),1,1, fc=[0.9, 0.9, 0.9, 0.6], label='Snow')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        return ax

    def visualize_grid(self, ax=None):
        """Create a visualization of the current fire state on the grid"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        if not (self.has_satellite_data and self.has_landscape_data):
            # Display a professional message when no data is available
            ax.text(0.5, 0.5, "Landscape data not available for this location\n\n"
                              "Possible reasons:\n"
                              "• Google Earth Engine not configured\n"
                              "• Location outside satellite coverage\n"
                              "• Cloud cover obscuring the area\n"
                              "• Recent imagery not available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red',
                    bbox=dict(boxstyle="round,pad=1", facecolor="lightyellow", alpha=0.8))
            ax.set_title("Landscape Data Unavailable")
            ax.axis('off')
            return ax
        
        # Display the landscape with NDVI as base
        ndvi_display = np.clip(self.landscape_data['ndvi'], 0, 1)
        
        # Overlay fire on top
        display_img = np.zeros((self.size, self.size, 3))
        
        # Green base for vegetation
        display_img[:, :, 1] = ndvi_display * 0.7
        display_img[:, :, 0] = ndvi_display * 0.1
        display_img[:, :, 2] = ndvi_display * 0.1
        
        # Add water bodies (blue)
        water_mask = self.landscape_data['water_mask'] > 0.5
        display_img[water_mask] = [0.2, 0.4, 0.8]
        
        # Add urban areas (gray)
        urban_mask = self.landscape_data['urban_mask'] > 0.5
        display_img[urban_mask] = [0.6, 0.6, 0.6]
        
        # Add snow areas (white)
        snow_mask = self.landscape_data['snow_mask'] > 0.5
        display_img[snow_mask] = [0.9, 0.9, 0.9]
        
        # Add fire colors
        fire_mask = self.grid > 0.7
        fire_intensity = np.clip(self.grid[fire_mask], 0.7, 2.0)
        
        # Color mapping for fire
        for i, intensity in enumerate(fire_intensity):
            if intensity < 1.0:  # Early fire (yellow to orange)
                r, g, b = 1.0, 1.0 - (intensity - 0.7) * 3.33, 0.0
            elif intensity < 1.5:  # Medium fire (orange to red)
                r, g, b = 1.0, 0.5 - (intensity - 1.0) * 1.0, 0.0
            else:  # Burned out (brown)
                r, g, b = 0.5, 0.2, 0.1
                
            # Get coordinates
            coords = np.where(fire_mask)
            display_img[coords[0][i], coords[1][i], 0] = r
            display_img[coords[0][i], coords[1][i], 1] = g
            display_img[coords[0][i], coords[1][i], 2] = b
        
        ax.imshow(display_img)
        ax.set_title(f"Grid-Based Fire Simulation\nTime: {self.time_elapsed} hours")
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, fc=[0.1, 0.5, 0.1], label='Vegetation'),
            plt.Rectangle((0,0),1,1, fc=[1.0, 0.8, 0.2], label='Ignition'),
            plt.Rectangle((0,0),1,1, fc=[1.0, 0.5, 0.0], label='Burning'),
            plt.Rectangle((0,0),1,1, fc=[0.5, 0.2, 0.1], label='Burned'),
            plt.Rectangle((0,0),1,1, fc=[0.2, 0.4, 0.8], label='Water'),
            plt.Rectangle((0,0),1,1, fc=[0.6, 0.6, 0.6], label='Urban'),
            plt.Rectangle((0,0),1,1, fc=[0.9, 0.9, 0.9], label='Snow')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
        
        return ax

# --- UI Layout ---
# Create a container to center the title and subtitle
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">Gaia\'s Prophecy🔥🌲</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Forest Fire Prediction and Simulation System    </p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Manual Coordinate Input Section ---
st.markdown('<div class="sub-header">Or Enter Coordinates Manually</div>', unsafe_allow_html=True)
manual_col1, manual_col2, manual_col3 = st.columns([1, 1, 2])
with manual_col1:
    manual_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=20.5937, format="%.6f")
with manual_col2:
    manual_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=78.9629, format="%.6f")
with manual_col3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Use These Coordinates", use_container_width=True):
        st.session_state['selected_point'] = {
            'lat': manual_lat,
            'lon': manual_lon
        }
        st.rerun()

# --- Map Section (Top Landscape Area) ---
st.markdown('<div class="sub-header">Select Location on Map</div>', unsafe_allow_html=True)
m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
m.add_child(folium.LatLngPopup())

# If we have a selected point, center the map on it
if 'selected_point' in st.session_state:
    point = st.session_state['selected_point']
    m.location = [point['lat'], point['lon']]
    m.zoom_start = 10
    # Add a marker for the selected point
    folium.Marker(
        [point['lat'], point['lon']],
        popup=f"Selected Location: {point['lat']:.6f}, {point['lon']:.6f}"
    ).add_to(m)

map_data = st_folium(m, height=400, use_container_width=True, key="map")

if map_data and map_data.get('last_clicked'):
    st.session_state['selected_point'] = {
        'lat': map_data['last_clicked']['lat'],
        'lon': map_data['last_clicked']['lng']
    }
    st.rerun()

# --- Initialize Services ---
gee_initialized = initialize_gee()
model, scaler, feature_names = load_model()

# --- Prediction and Simulation Options ---
if 'selected_point' in st.session_state:
    point = st.session_state['selected_point']
    
    # Display selected coordinates
    coord_col1, coord_col2 = st.columns(2)
    with coord_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latitude</div>
            <div class="metric-value">{point['lat']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with coord_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Longitude</div>
            <div class="metric-value">{point['lon']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction and Simulation Tabs
    pred_tab, sim_tab = st.tabs(["🔥 Risk Prediction", "💨 Advanced Simulation"])
    
    with pred_tab:
        if st.button("Predict Fire Risk", use_container_width=True, type="primary"):
            with st.spinner("Fetching weather, satellite, and rainfall data..."):
                # Get all available data
                weather_data = get_live_weather_data(point['lat'], point['lon'])
                gee_data = get_gee_data(point['lat'], point['lon'])
                rainfall_info = get_precipitation_data(point['lat'], point['lon'])                 
                
                if weather_data or gee_data:
                    # Adapt features with real rainfall data
                    model_features = adapt_features_to_model(weather_data, gee_data, 
                                                           point['lat'], point['lon'])
                    
                    # Ensure we have all required features
                    if feature_names:
                        feature_vector = [model_features[name] for name in feature_names]
                        
                        try:
                            # Scale the features
                            scaled_features = scaler.transform([feature_vector])
                            
                            # Make prediction
                            prediction_prob = model.predict_proba(scaled_features)[0][1]
                            
                            # Display results
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Fire Risk Probability</div>
                                <div class="metric-value">{prediction_prob:.2%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Risk assessment
                            if prediction_prob > 0.7:
                                st.markdown('<p class="risk-high">🚨 High Risk Area - Immediate attention required</p>', unsafe_allow_html=True)
                            elif prediction_prob > 0.4:
                                st.markdown('<p class="risk-medium">⚠️ Moderate Risk Area - Monitor closely</p>', unsafe_allow_html=True)
                            else:
                                st.markdown('<p class="risk-low">✅ Low Risk Area - Normal conditions</p>', unsafe_allow_html=True)
                                
                            # Show feature values
                            with st.expander("Show Detailed Feature Values", expanded=False):
                                formatted_features = format_feature_names(model_features)
                                if formatted_features:
                                    feature_df = pd.DataFrame.from_dict(formatted_features, orient='index', columns=['Value'])
                                    st.dataframe(feature_df, use_container_width=True)
                                else:
                                    st.write("No feature data available")
                                    
                            # Add explanation of feature mapping
                            with st.expander("How features are calculated", expanded=False):
                                st.info("""
                                **Feature Mapping Explanation:**
                                - **X, Y**: Mapped from coordinates
                                - **FFMC**: Based on temperature and vegetation (higher temp → lower moisture → higher FFMC)
                                - **DMC**: Based on vegetation density (lower vegetation → higher duff moisture)
                                - **DC**: Long-term drought effect influenced by vegetation
                                - **ISI**: Based on wind speed and temperature
                                - **temp, RH, wind**: From live weather data
                                - **rain**: Real rainfall data from Open-Meteo API (24h average)
                                """)
                                    
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
                    else:
                        st.error("Feature names not available for prediction.")
                else:
                    st.markdown("""
                    <div class="satellite-error">
                        <h3>⚠️ Satellite Data Unavailable</h3>
                        <p>Could not fetch satellite data for this location. This could be due to:</p>
                        <ul>
                            <li>Google Earth Engine not being properly configured</li>
                            <li>Location outside satellite coverage area</li>
                            <li>Cloud cover obscuring the area</li>
                            <li>Recent satellite imagery not available</li>
                        </ul>
                        <p>Please try a different location or ensure Google Earth Engine is properly set up.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with sim_tab:
        st.markdown('<div class="sub-header">Simulation Parameters</div>', unsafe_allow_html=True)
        
        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            st.markdown('<div class="simulation-control"><h4>Area Settings</h4>', unsafe_allow_html=True)
            sim_size = st.slider("Simulation Area Size", 50, 200, 100, help="Larger areas take longer to simulate")
            sim_hours = st.slider("Simulation Duration (hours)", 1, 72, 24)
            st.markdown('</div>', unsafe_allow_html=True)
        with sim_col2:
            st.markdown('<div class="simulation-control"><h4>Time Settings</h4>', unsafe_allow_html=True)
            time_step = st.slider("Time Step (hours)", 1, 4, 2, help="Smaller steps are more accurate but slower")
            sim_speed = st.slider("Animation Speed", 0.1, 2.0, 0.5, help="Speed of visualization updates")
            st.markdown('</div>', unsafe_allow_html=True)
        with sim_col3:
            st.markdown('<div class="simulation-control"><h4>Weather Settings</h4>', unsafe_allow_html=True)
            wind_override = st.checkbox("Override Weather", help="Use custom weather for simulation")
            if wind_override:
                custom_wind = st.slider("Wind Speed (m/s)", 0.0, 20.0, 5.0)
                custom_dir = st.slider("Wind Direction (degrees)", 0, 360, 90)
                custom_humidity = st.slider("Humidity (%)", 10, 100, 45)
                custom_temp = st.slider("Temperature (°C)", 0, 40, 25)
                custom_rain = st.slider("Recent Rainfall (mm)", 0.0, 50.0, 0.0)
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Run Advanced Fire Simulation", use_container_width=True, type="primary"):
            with st.spinner("Initializing advanced simulation..."):
                # Get weather data
                if wind_override:
                    weather_data = {
                        'wind_speed': custom_wind,
                        'wind_deg': custom_dir,
                        'humidity': custom_humidity,
                        'temp': custom_temp,
                        'rain_24h': custom_rain
                    }
                else:
                    weather_data = get_live_weather_data(point['lat'], point['lon'])
                    if not weather_data:
                        weather_data = {
                            'wind_speed': 5.0,
                            'wind_deg': 90,
                            'humidity': 45,
                            'temp': 25,
                            'rain_24h': 0
                        }
                    # Add recent rainfall data
                    precipitation_data = get_precipitation_data(point['lat'], point['lon'])
                    weather_data['rain_24h'] = precipitation_data['avg_24h_rain_mm']
                
                # Initialize simulator
                simulator = EnhancedFireSimulator(point['lat'], point['lon'], size=sim_size)
                
                # Check if we have data for simulation
                if not simulator.has_satellite_data or not simulator.has_landscape_data:
                    st.markdown("""
                    <div class="satellite-error">
                        <h3>⚠️ Satellite Data Unavailable</h3>
                        <p>Fire simulation cannot be displayed for this location because satellite imagery is not available.</p>
                        <p>Possible reasons:</p>
                        <ul>
                            <li>Google Earth Engine not properly configured</li>
                            <li>Location outside satellite coverage area</li>
                            <li>Cloud cover obscuring the area</li>
                            <li>Recent satellite imagery not available</li>
                        </ul>
                        <p>Please try a different location or ensure Google Earth Engine is properly set up.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()
                
            st.markdown('<div class="sub-header">Simulation Results</div>', unsafe_allow_html=True)
            
            # Create layout for simulation visualizations
            vis_col1, vis_col2 = st.columns(2)
            
            with vis_col1:
                st.markdown("**🌍 Map-Based Visualization**")
                # Placeholder for map simulation visualization
                map_placeholder = st.empty()
                # Create figure once and reuse it
                fig_map, ax_map = plt.subplots(figsize=(6, 6))
                
            with vis_col2:
                st.markdown("**🔲 Grid-Based Visualization**")
                # Placeholder for grid simulation visualization
                grid_placeholder = st.empty()
                # Create figure once and reuse it
                fig_grid, ax_grid = plt.subplots(figsize=(6, 6))
            
            # Statistics and progress chart below visualizations
            stats_col, chart_col = st.columns([0.4, 0.6])
            
            with stats_col:
                st.markdown("**📊 Statistics**")
                # Placeholder for statistics
                stats_placeholder = st.empty()
            
            with chart_col:
                st.markdown("**📈 Progress Chart**")
                # Placeholder for progress chart
                chart_placeholder = st.empty()
                # Create figure for chart once
                fig2, ax2 = plt.subplots(figsize=(8, 4))
            
            # Progress bar
            st.markdown("**Progress**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run simulation
            for hour in range(0, sim_hours + 1, int(time_step)):
                # Update simulation
                current_grid = simulator.step(weather_data, time_step)
                
                # Update map visualization - clear and reuse the same axes
                ax_map.clear()
                simulator.visualize_on_map(ax_map)
                map_placeholder.pyplot(fig_map)
                
                # Update grid visualization - clear and reuse the same axes
                ax_grid.clear()
                simulator.visualize_grid(ax_grid)
                grid_placeholder.pyplot(fig_grid)
                
                # Update statistics
                stats = simulator.get_stats()
                stats_df = pd.DataFrame({
                    'Metric': ['Time Elapsed', 'Burning Area', 'Burned Area', 'Total Affected', 
                              'Fire Line Length', 'Rate of Spread', '% Burned'],
                    'Value': [
                        f"{stats['time_elapsed']} hours",
                        f"{stats['burning_area']:.2f} ha",
                        f"{stats['burned_area']:.2f} ha",
                        f"{stats['total_affected']} cells",
                        f"{stats['fire_line_length']:.0f} m",
                        f"{stats['rate_of_spread']:.2f} ha/hr",
                        f"{stats['percent_burned']:.1f}%"
                    ]
                })
                stats_placeholder.table(stats_df)
                
                # Update history chart - clear and reuse the same axes
                if len(simulator.history) > 1:
                    history_df = pd.DataFrame(simulator.history)
                    ax2.clear()
                    ax2.plot(history_df['time'], history_df['burning'], 'r-', label='Burning')
                    ax2.plot(history_df['time'], history_df['burned'], 'b-', label='Burned')
                    ax2.set_xlabel('Time (hours)')
                    ax2.set_ylabel('Area (cells)')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    chart_placeholder.pyplot(fig2)
                
                # Update progress
                progress = min(1.0, hour / sim_hours)
                progress_bar.progress(progress)
                status_text.text(f"Simulating... {hour}/{sim_hours} hours completed")
                
                # Control speed
                time.sleep(1 / sim_speed)
            
            status_text.text("Simulation complete!")
            st.success("Fire simulation finished.")
            
            # Close figures to free memory
            plt.close(fig_map)
            plt.close(fig_grid)
            plt.close(fig2)
            
            # Download results
            if len(simulator.history) > 0:
                history_df = pd.DataFrame(simulator.history)
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download Simulation Data",
                    data=csv,
                    file_name="fire_simulation_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
else:
    st.info("Please click on the map to select a location for analysis.")

# Add cleanup function to close all figures when the app is done
plt.close('all')