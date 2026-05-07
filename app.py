"""
TERRAVOLT INTELLIGENCE
AI-Powered Geospatial Renewable Energy Optimization Platform
MVP: Solar + Wind scoring for any location on Earth
"""

from flask import Flask, render_template, request, jsonify
import requests
import math
import json
from datetime import datetime, timedelta

app = Flask(__name__)

# ================================================================
# SOLAR SCORE CALCULATION (NASA POWER API)
# ================================================================

def get_solar_score(lat, lon):
    """
    Calculate solar energy potential for a given location.
    Uses NASA POWER API for solar irradiance data.
    Returns score 0-100.
    """
    try:
        # NASA POWER API endpoint
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point"
        
        params = {
            'parameters': 'ALLSKY_SFC_SW_DWN',
            'community': 'RE',
            'longitude': lon,
            'latitude': lat,
            'start': '20230101',
            'end': '20231231',
            'format': 'JSON'
        }
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        # Extract solar irradiance values (W/m²)
        solar_data = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
        values = list(solar_data.values())
        
        if not values:
            return 50  # Default mid score if no data
        
        # Calculate average daily solar irradiance
        avg_irradiance = sum(values) / len(values)
        
        # Convert to score (0-100)
        # Typical range: 50-300 W/m²
        # 300+ W/m² = 100 score
        # 50 W/m² = 0 score
        score = min(100, max(0, (avg_irradiance - 50) / 250 * 100))
        
        return round(score, 1), avg_irradiance
        
    except Exception as e:
        print(f"Solar API error: {e}")
        return 50, 150  # Default fallback


# ================================================================
# WIND SCORE CALCULATION (Open-Meteo API)
# ================================================================

def get_wind_score(lat, lon):
    """
    Calculate wind energy potential for a given location.
    Uses Open-Meteo API for wind speed data.
    Returns score 0-100.
    """
    try:
        url = f"https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'hourly': 'wind_speed_10m',
            'timezone': 'auto'
        }
        
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        if 'hourly' not in data:
            return 50, 5  # Default fallback
        
        wind_speeds = data['hourly']['wind_speed_10m']
        valid_speeds = [s for s in wind_speeds if s is not None]
        
        if not valid_speeds:
            return 50, 5
        
        avg_wind_speed = sum(valid_speeds) / len(valid_speeds)
        
        # Convert to score (0-100)
        # Wind turbines need 4-5 m/s minimum
        # 10+ m/s = 100 score
        # 0 m/s = 0 score
        score = min(100, max(0, (avg_wind_speed - 3) / 12 * 100))
        
        return round(score, 1), avg_wind_speed
        
    except Exception as e:
        print(f"Wind API error: {e}")
        return 50, 5


# ================================================================
# HYBRID SCORE (Weighted combination)
# ================================================================

def get_hybrid_score(solar_score, wind_score, solar_weight=0.5, wind_weight=0.5):
    """
    Calculate hybrid renewable energy score.
    Default: 50% solar, 50% wind
    """
    hybrid = (solar_score * solar_weight) + (wind_score * wind_weight)
    return round(hybrid, 1)


# ================================================================
# LOCATION NAME LOOKUP (Reverse geocoding)
# ================================================================

def get_location_name(lat, lon):
    """Get location name from coordinates using Nominatim API"""
    try:
        url = f"https://nominatim.openstreetmap.org/reverse"
        params = {
            'lat': lat,
            'lon': lon,
            'format': 'json',
            'zoom': 10
        }
        headers = {
            'User-Agent': 'TerraVolt-Intelligence/1.0'
        }
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if 'display_name' in data:
            # Extract city/region from full address
            parts = data['display_name'].split(',')
            if len(parts) >= 2:
                return parts[0].strip()
            return parts[0].strip()
        return f"{lat:.2f}, {lon:.2f}"
    except:
        return f"{lat:.2f}, {lon:.2f}"


# ================================================================
# TOP LOCATIONS (Precomputed best renewable energy sites)
# ================================================================

def get_top_locations(energy_type='hybrid', limit=10):
    """
    Return top locations for renewable energy.
    Based on known ideal locations for solar/wind.
    """
    # Known optimal locations for renewable energy
    locations = [
        {'name': 'Sahara Desert, Africa', 'lat': 23.0, 'lon': 13.0, 'solar': 98, 'wind': 65},
        {'name': 'Atacama Desert, Chile', 'lat': -24.5, 'lon': -69.0, 'solar': 97, 'wind': 60},
        {'name': 'Gobi Desert, Asia', 'lat': 42.0, 'lon': 105.0, 'solar': 92, 'wind': 70},
        {'name': 'North Sea, Europe', 'lat': 55.0, 'lon': 3.0, 'solar': 55, 'wind': 92},
        {'name': 'Patagonia, Argentina', 'lat': -45.0, 'lon': -68.0, 'solar': 85, 'wind': 88},
        {'name': 'Great Plains, USA', 'lat': 40.0, 'lon': -100.0, 'solar': 88, 'wind': 85},
        {'name': 'Thar Desert, India', 'lat': 27.0, 'lon': 71.0, 'solar': 94, 'wind': 60},
        {'name': 'Australian Outback', 'lat': -25.0, 'lon': 135.0, 'solar': 96, 'wind': 55},
        {'name': 'Mojave Desert, USA', 'lat': 35.0, 'lon': -115.0, 'solar': 95, 'wind': 60},
        {'name': 'South Africa Coast', 'lat': -34.0, 'lon': 22.0, 'solar': 88, 'wind': 82},
        {'name': 'Mongolia Steppe', 'lat': 46.0, 'lon': 105.0, 'solar': 85, 'wind': 85},
        {'name': 'Iceland', 'lat': 64.0, 'lon': -19.0, 'solar': 35, 'wind': 90},
        {'name': 'British Coast, UK', 'lat': 55.0, 'lon': -5.0, 'solar': 45, 'wind': 88},
        {'name': 'Horn of Africa', 'lat': 8.0, 'lon': 48.0, 'solar': 92, 'wind': 75},
        {'name': 'Central Australia', 'lat': -22.0, 'lon': 140.0, 'solar': 95, 'wind': 60},
    ]
    
    # Calculate hybrid scores
    for loc in locations:
        loc['hybrid'] = round((loc['solar'] + loc['wind']) / 2, 1)
    
    # Sort by selected energy type
    if energy_type == 'solar':
        locations.sort(key=lambda x: x['solar'], reverse=True)
    elif energy_type == 'wind':
        locations.sort(key=lambda x: x['wind'], reverse=True)
    else:
        locations.sort(key=lambda x: x['hybrid'], reverse=True)
    
    return locations[:limit]


# ================================================================
# FLASK ROUTES
# ================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze a location for renewable energy potential"""
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    
    if lat is None or lon is None:
        return jsonify({'error': 'Missing coordinates'}), 400
    
    # Get location name
    location_name = get_location_name(lat, lon)
    
    # Get solar score
    solar_score, solar_irradiance = get_solar_score(lat, lon)
    
    # Get wind score
    wind_score, wind_speed = get_wind_score(lat, lon)
    
    # Calculate hybrid score
    hybrid_score = get_hybrid_score(solar_score, wind_score)
    
    # Determine recommendation
    if hybrid_score >= 80:
        recommendation = "Excellent location for renewable energy!"
    elif hybrid_score >= 65:
        recommendation = "Good potential for renewable energy."
    elif hybrid_score >= 50:
        recommendation = "Moderate potential. Consider hybrid system."
    else:
        recommendation = "Limited potential. May not be economically viable."
    
    return jsonify({
        'location': location_name,
        'lat': lat,
        'lon': lon,
        'solar_score': solar_score,
        'solar_irradiance': round(solar_irradiance, 1),
        'wind_score': wind_score,
        'wind_speed': round(wind_speed, 1),
        'hybrid_score': hybrid_score,
        'recommendation': recommendation
    })


@app.route('/api/top', methods=['GET'])
def top_locations():
    """Get top locations for renewable energy"""
    energy_type = request.args.get('type', 'hybrid')
    limit = int(request.args.get('limit', 10))
    
    locations = get_top_locations(energy_type, limit)
    
    return jsonify({
        'energy_type': energy_type,
        'count': len(locations),
        'locations': locations
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'operational',
        'version': '1.0.0',
        'name': 'TerraVolt Intelligence'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🌍 TERRAVOLT INTELLIGENCE")
    print("=" * 60)
    print("\n📍 Server running at: http://127.0.0.1:5000")
    print("📡 API endpoints:")
    print("   POST /api/analyze - Analyze location")
    print("   GET  /api/top     - Top locations")
    print("   GET  /api/health  - Health check")
    print("\nPress Ctrl+C to stop\n")
    app.run(debug=True, port=5000)