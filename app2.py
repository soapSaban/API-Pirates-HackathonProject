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
image_base64 = get_base64_of_image("1817788.jpg")
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
    
    /* ... (other CSS rules) ... */
    
    /* Add this new rule to center align values in dataframe cells */
    .stDataFrame [data-testid="stDataFrame"] td {{
        text-align: center !important;
    }}
    
    /* Ensure the column headers are also centered for consistency */
    .stDataFrame [data-testid="stDataFrame"] th {{
        text-align: center !important;
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
        color: #7f8c8d;
        margin: 0 auto 1.5rem auto;
        width: 100%;
    }}
    
    .sub-header {{
        font-size: 1.4rem;
        color: #2c3e50;
        border-bottom: 2px solid #ff4b4b;
        padding-bottom: 0.3rem;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
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
        return False

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model and scaler."""
    try:
        model = joblib.load('model/fire_prediction_model.joblib')
        scaler = joblib.load('model/feature_scaler.joblib')
        feature_names = joblib.load('model/feature_names.joblib')
        return model, scaler, feature_names
    except FileNotFoundError:
        return None, None, None
    
# --- Data Fetching Functions ---
def get_live_weather_data(lat, lon):
    """Fetches live weather data from OpenWeatherMap."""
    try:
        if not OPENWEATHER_API_KEY:
            return None
            
        url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            'temp': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'wind_speed': data['wind']['speed'],
            'wind_deg': data['wind'].get('deg', 0) # Use .get for safety
        }
    except Exception as e:
        return None

def format_feature_names(features_dict):
    """
    Converts raw feature names and values to human-readable format with proper units.
    """
    if not features_dict:
        return None
    
    # Mapping of feature names to human-readable names and units
    feature_mapping = {
        'temp': {'name': 'Temperature', 'unit': '°C', 'format': '{:.1f}'},
        'humidity': {'name': 'Humidity', 'unit': '%', 'format': '{:.0f}'},
        'wind_speed': {'name': 'Wind Speed', 'unit': 'm/s', 'format': '{:.2f}'},
        'wind_deg': {'name': 'Wind Direction', 'unit': 'degrees', 'format': '{:.0f}'},
        'aspect': {'name': 'Aspect', 'unit': 'degrees', 'format': '{:.1f}'},
        'lst': {'name': 'Land Surface Temperature', 'unit': '°C', 'format': '{:.1f}'},
        'ndvi': {'name': 'NDVI (Vegetation Index)', 'unit': '', 'format': '{:.3f}'},
        'slope': {'name': 'Slope', 'unit': 'degrees', 'format': '{:.1f}'}
    }
    
    formatted_features = {}
    
    for key, value in features_dict.items():
        if key in feature_mapping and value is not None:
            mapping = feature_mapping[key]
            try:
                formatted_value = mapping['format'].format(value)
                formatted_name = f"{mapping['name']} ({mapping['unit']})" if mapping['unit'] else mapping['name']
                formatted_features[formatted_name] = formatted_value
            except (ValueError, TypeError):
                # If formatting fails, use the original value
                formatted_name = f"{mapping['name']} ({mapping['unit']})" if mapping['unit'] else mapping['name']
                formatted_features[formatted_name] = str(value)
        else:
            # Keep unknown features as is
            formatted_features[key] = value
    
    return formatted_features

def get_gee_data(lat, lon):
    """Fetches landscape data from Google Earth Engine for a given point."""
    try:
        # Check if GEE is initialized
        if not ee.data._initialized:
            return None
            
        point = ee.Geometry.Point(lon, lat)
        
        # Get current date and date 30 days ago
        current_date = ee.Date(time.time() * 1000)
        start_date = current_date.advance(-30, 'day')
        
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

        dem = ee.Image('USGS/SRTMGL1_003')
        slope = ee.Terrain.slope(dem).rename('slope')
        aspect = ee.Terrain.aspect(dem).rename('aspect')

        feature_stack = ee.Image.cat([ndvi, lst, slope, aspect])
        feature_values = feature_stack.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=100
        ).getInfo()
        
        # Ensure all expected features are present
        expected_features = ['ndvi', 'lst', 'slope', 'aspect']
        for feature in expected_features:
            if feature not in feature_values:
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
        
        # Get landscape data
        self.landscape_data = self._get_landscape_data()
        
        # Only initialize fire if we have data
        self.has_data = self.satellite_image is not None and self.landscape_data is not None
        
        if self.has_data:
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
        self.burning_cells = np.sum(self.grid >= 0.8) if self.has_data else 0
        self.time_elapsed = 0
        self.history = []

    def _get_landscape_data(self):
        # Return None if no satellite image is available
        if self.satellite_image is None:
            return None
            
        # Create more realistic landscape data
        try:
            # Create realistic slope with some variation
            x, y = np.meshgrid(np.linspace(-1, 1, self.size), np.linspace(-1, 1, self.size))
            
            # Generate multiple hills/valleys for realistic topography
            slope_grid = (np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) + 
                        0.5 * np.sin(4 * np.pi * x) * np.cos(4 * np.pi * y)) * 10 + 5
            slope_grid = np.clip(slope_grid, 0, 20)  # Limit slope to 0-20 degrees
            
            # Create realistic vegetation pattern based on topography
            # Higher vegetation in valleys, less on steep slopes
            ndvi_grid = 0.7 * (1 - 0.05 * slope_grid) + 0.1 * np.random.randn(self.size, self.size)
            ndvi_grid = np.clip(ndvi_grid, 0.1, 0.9)  # Keep NDVI in reasonable range
            
            # Add some random patches of different vegetation
            for _ in range(5):
                i, j = np.random.randint(0, self.size, 2)
                size = np.random.randint(5, 15)
                value = np.random.uniform(-0.2, 0.2)
                ndvi_grid[max(0, i-size):min(self.size, i+size), 
                         max(0, j-size):min(self.size, j+size)] += value
            
            ndvi_grid = np.clip(ndvi_grid, 0.1, 0.9)
            
            return {'slope': slope_grid, 'ndvi': ndvi_grid}
        
        except Exception as e:
            return None

    def _create_fuel_map(self):
        """Create a fuel map based on NDVI values"""
        if not self.has_data:
            return np.zeros((self.size, self.size))
            
        # Convert NDVI to fuel load (kg/m²)
        # Higher NDVI = more fuel
        fuel_map = 0.5 + 2.0 * self.landscape_data['ndvi']
        
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

    def _calculate_spread_probability(self, r, c, weather):
        """Calculate probability of fire spreading to cell (r, c)"""
        if not self.has_data:
            return 0  # No spread without data
            
        # Base probability
        prob = 0.15
        
        # Fuel effect (more fuel = higher probability)
        fuel_effect = 0.3 * (self.fuel_map[r, c] / np.max(self.fuel_map))
        prob += fuel_effect
        
        # Slope effect (fire spreads faster uphill)
        if r > 0 and c > 0 and r < self.size-1 and c < self.size-1:
            slope_effect = 0.002 * (self.landscape_data['slope'][r, c] - 
                                  np.mean(self.landscape_data['slope'][r-1:r+2, c-1:c+2]))
            prob += slope_effect
        
        # Wind effect
        wind_dir = weather.get('wind_deg', 0)
        wind_speed = weather.get('wind_speed', 0)
        
        # Calculate wind direction components
        wind_rad = np.radians(wind_dir)
        wind_u = -wind_speed * np.sin(wind_rad)  # u component (east-west)
        wind_v = -wind_speed * np.cos(wind_rad)  # v component (north-south)
        
        # Normalize wind components for direction bias
        if wind_speed > 0:
            # Calculate direction from burning cell to target cell
            # This is simplified - in a real model, we'd track fire front properly
            wind_effect = 0.2 * (wind_speed / 10)
            prob += wind_effect
        
        # Humidity effect (lower humidity = higher probability)
        humidity = weather.get('humidity', 50)
        humidity_effect = 0.003 * (100 - humidity)
        prob += humidity_effect
        
        # Temperature effect (higher temperature = higher probability)
        temperature = weather.get('temp', 20)
        temp_effect = 0.002 * (temperature - 20)
        prob += temp_effect
        
        return np.clip(prob, 0.05, 0.8)

    def step(self, weather, time_step=1):
        """Advance the simulation by one time step"""
        if not self.has_data:
            # Return empty grid if no data is available
            return self.grid
            
        # Update current burning cells
        burning_mask = self.grid >= 0.8
        self.grid[burning_mask] += 0.1 * time_step  # Increase fire intensity
        
        # Mark cells that have burned out
        burned_out = self.grid > 1.5
        self.grid[burned_out] = 2.0  # Completely burned
        
        # Find newly ignited cells
        new_fires = np.zeros_like(self.grid, dtype=bool)
        
        # Find all burning cells
        burning_cells = np.argwhere(self.grid >= 0.8)
        
        for r, c in burning_cells:
            # Check all 8 neighboring cells
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    
                    nr, nc = r + dr, c + dc
                    
                    # Check if within bounds and not already burning/burned
                    if (0 <= nr < self.size and 0 <= nc < self.size and 
                        self.grid[nr, nc] < 0.8):
                        
                        # Calculate spread probability
                        spread_prob = self._calculate_spread_probability(nr, nc, weather)
                        
                        # Apply probability
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
            'total_affected': self.burning_cells + self.burned_cells
        })
        
        return self.grid

    def get_stats(self):
        """Get current simulation statistics"""
        total_area = self.size * self.size * (self.scale ** 2) / 10000  # Convert to hectares
        return {
            'time_elapsed': self.time_elapsed,
            'burning_cells': self.burning_cells,
            'burned_cells': self.burned_cells,
            'total_affected': self.burning_cells + self.burned_cells,
            'burning_area': self.burning_cells * (self.scale ** 2) / 10000,  # hectares
            'burned_area': self.burned_cells * (self.scale ** 2) / 10000,    # hectares
            'percent_burned': 100 * self.burned_cells / (self.size * self.size),
            'has_data': self.has_data
        }

    def visualize_on_map(self, ax=None):
        """Create a visualization of the current fire state overlaid on the satellite image"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        if not self.has_data:
            # Display a message when no data is available
            ax.text(0.5, 0.5, "No satellite data available\nfor this location", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14, color='red')
            ax.set_title("No Data Available")
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
        
        return ax

    def visualize_grid(self, ax=None):
        """Create a visualization of the current fire state on the grid"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        if not self.has_data:
            # Display a message when no data is available
            ax.text(0.5, 0.5, "No landscape data available\nfor this location", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14, color='red')
            ax.set_title("No Data Available")
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

# Show warning if GEE is not initialized but allow the app to continue
if not gee_initialized:
    st.warning("⚠️ Google Earth Engine is not initialized. The app will use mock data for demonstration.")

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
            with st.spinner("Fetching data and calculating risk..."):
                weather_features = get_live_weather_data(point['lat'], point['lon'])
                gee_features = get_gee_data(point['lat'], point['lon'])
                
                if weather_features and gee_features:
                    all_features = {**weather_features, **gee_features}
                    
                    # Ensure we have all required features
                    if feature_names:
                        feature_vector = [all_features.get(name, 0) for name in feature_names]
                        
                        try:
                            scaled_features = scaler.transform([feature_vector])
                            prediction_prob = model.predict_proba(scaled_features)[0][1]
                            
                            # Display risk with appropriate styling
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Fire Risk Probability</div>
                                <div class="metric-value">{prediction_prob:.2%}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if prediction_prob > 0.7:
                                st.markdown('<p class="risk-high">🚨 High Risk Area - Immediate attention required</p>', unsafe_allow_html=True)
                            elif prediction_prob > 0.4:
                                st.markdown('<p class="risk-medium">⚠️ Moderate Risk Area - Monitor closely</p>', unsafe_allow_html=True)
                            else:
                                st.markdown('<p class="risk-low">✅ Low Risk Area - Normal conditions</p>', unsafe_allow_html=True)
                                
                            # Show feature values for debugging
                            with st.expander("Show Detailed Feature Values", expanded=False):
                                # Format the feature names and values
                                formatted_features = format_feature_names(all_features)
                                if formatted_features:
                                    feature_df = pd.DataFrame.from_dict(formatted_features, orient='index', columns=['Value'])
                                    st.dataframe(feature_df, use_container_width=True)
                                else:
                                    st.write("No feature data available")
                                
                        except Exception as e:
                            st.error(f"Error making prediction: {e}")
                    else:
                        st.error("Feature names not available for prediction.")
                else:
                    st.error("Could not fetch all required data for prediction.")
    
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
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Run Advanced Fire Simulation", use_container_width=True, type="primary"):
            with st.spinner("Initializing advanced simulation..."):
                # Get weather data
                if wind_override:
                    weather_data = {
                        'wind_speed': custom_wind,
                        'wind_deg': custom_dir,
                        'humidity': custom_humidity,
                        'temp': custom_temp
                    }
                else:
                    weather_data = get_live_weather_data(point['lat'], point['lon'])
                    if not weather_data:
                        weather_data = {
                            'wind_speed': 5.0,
                            'wind_deg': 90,
                            'humidity': 45,
                            'temp': 25
                        }
                
                # Initialize simulator
                simulator = EnhancedFireSimulator(point['lat'], point['lon'], size=sim_size)
                
                # Check if we have data for simulation
                if not simulator.has_data:
                    st.markdown("""
                    <div class="no-data-warning">
                        <h3>⚠️ No Satellite Data Available</h3>
                        <p>Fire simulation cannot be displayed for this location because satellite imagery is not available.</p>
                        <p>Please try a different location or check if Google Earth Engine is properly configured.</p>
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
                    'Metric': ['Time Elapsed', 'Burning Area', 'Burned Area', 'Total Affected', '% Burned'],
                    'Value': [
                        f"{stats['time_elapsed']} hours",
                        f"{stats['burning_area']:.2f} ha",
                        f"{stats['burned_area']:.2f} ha",
                        f"{stats['total_affected']} cells",
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