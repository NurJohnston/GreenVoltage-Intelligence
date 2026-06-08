"""
TERRAVOLT INTELLIGENCE - SOUTH AFRICA EDITION
AI-Powered Geospatial Renewable Energy Optimization Platform
With LIVE data + ML fallback + Caching
"""

from flask import Flask, render_template, request, jsonify
from flask_caching import Cache
import requests
import math
import numpy as np
from datetime import datetime, timedelta
from ml_model import RenewableEnergyPredictor
import hashlib
import json

app = Flask(__name__)

# Setup caching to avoid redundant API calls
app.config['CACHE_TYPE'] = 'SimpleCache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 3600  # 1 hour cache
cache = Cache(app)

# Initialize ML predictor
print("Loading ML models...")
ml_predictor = RenewableEnergyPredictor()
ml_predictor.load_models()
print("ML models ready!")

# South Africa bounds
SA_BOUNDS = {
    'min_lat': -35.0,
    'max_lat': -22.0,
    'min_lon': 16.0,
    'max_lon': 33.0
}

# Cache keys for common locations
def get_cache_key(lat, lon, data_type='analysis'):
    return f"{data_type}:{lat:.2f}:{lon:.2f}"

# ================================================================
# SOLAR SCORE CALCULATION (NASA POWER API - LIVE)
# ================================================================

def get_solar_score(lat, lon):
    """Get live solar data from NASA POWER"""
    try:
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'parameters': 'ALLSKY_SFC_SW_DWN',
            'community': 'RE',
            'longitude': lon,
            'latitude': lat,
            'start': '20230101',
            'end': '20231231',
            'format': 'JSON'
        }
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if 'properties' not in data or 'parameter' not in data['properties']:
            return None, None
        
        solar_data = data['properties']['parameter'].get('ALLSKY_SFC_SW_DWN', {})
        values = [v for v in solar_data.values() if v is not None]
        
        if not values:
            return None, None
        
        avg_irradiance = sum(values) / len(values)
        avg_irradiance = avg_irradiance * 11.574  # convert MJ/m²/day to W/m²
        #score = min(100, max(0, (avg_irradiance - 50) / 250 * 100))
        #score = min(100, max(0, (avg_irradiance - 289) / 988 * 100))
        score = min(100, max(0, (avg_irradiance - 200) / 800 * 100))
        return round(score, 1), avg_irradiance
        
    except Exception as e:
        print(f"Solar API error: {e}")
        return None, None


# ================================================================
# WIND SCORE CALCULATION (Open-Meteo API - LIVE)
# ================================================================

def get_wind_score(lat, lon):
    """Get live wind data from Open-Meteo"""
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'hourly': 'wind_speed_10m',
            'wind_speed_unit': 'ms',
            'timezone': 'auto'
        }
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        if 'hourly' not in data:
            return None, None
        
        wind_speeds = data['hourly']['wind_speed_10m']
        valid_speeds = [s for s in wind_speeds if s is not None]
        
        if not valid_speeds:
            return None, None
        
        avg_wind_speed = sum(valid_speeds) / len(valid_speeds)
        #score = min(100, max(0, (avg_wind_speed - 3) / 12 * 100))
        score = min(100, max(0, (avg_wind_speed - 2.9) / (14.9 - 2.9) * 100))
        return round(score, 1), avg_wind_speed
        
    except Exception as e:
        print(f"Wind API error: {e}")
        return None, None


def get_hybrid_score(solar_score, wind_score):
    if solar_score is None or wind_score is None:
        return None
    return round((solar_score + wind_score) / 2, 1)


def get_location_name(lat, lon):
    """Reverse geocoding with fallback"""
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 10}
        headers = {'User-Agent': 'TerraVolt-Intelligence/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=8)
        data = response.json()
        if 'display_name' in data:
            parts = data['display_name'].split(',')
            return parts[0].strip()
        return f"{lat:.2f}, {lon:.2f}"
    except:
        return f"{lat:.2f}, {lon:.2f}"


# ================================================================
# ML-BASED HEATMAP & HOTSPOTS (NO HARDCODED ESTIMATES)
# ================================================================

def get_ml_score_for_grid(lat, lon, energy_type='hybrid'):
    """Get ML-predicted score for any location"""
    pred = ml_predictor.predict_with_confidence(lat, lon)
    
    if energy_type == 'solar':
        return pred['solar']
    elif energy_type == 'wind':
        return pred['wind']
    else:
        return pred['hybrid']


# ================================================================
# FLASK ROUTES
# ================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint - tries live API first, falls back to ML"""
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    
    if lat is None or lon is None:
        return jsonify({'error': 'Missing coordinates'}), 400
    
    # Check cache first
    cache_key = get_cache_key(lat, lon)
    cached = cache.get(cache_key)
    if cached:
        return jsonify(cached)
    
    location_name = get_location_name(lat, lon)
    
    # Try live APIs
    solar_score, solar_irradiance = get_solar_score(lat, lon)
    wind_score, wind_speed = get_wind_score(lat, lon)
    
    data_source = 'live'
    
    # If live APIs fail, use ML predictions
    if solar_score is None or wind_score is None:
        data_source = 'ml_fallback'
        ml_pred = ml_predictor.predict_with_confidence(lat, lon)
        solar_score = ml_pred['solar']
        wind_score = ml_pred['wind']
        solar_irradiance = ml_pred.get('solar_power_wm2', 150)
        wind_speed = ml_pred.get('wind_power_wm2', 5) ** (1/3)  # Approximate
    else:
        # Still get ML prediction for comparison
        ml_pred = ml_predictor.predict_with_confidence(lat, lon)
    
    hybrid_score = get_hybrid_score(solar_score, wind_score)
    
    if hybrid_score and hybrid_score >= 80:
        recommendation = "Excellent location for renewable energy in South Africa!"
    elif hybrid_score and hybrid_score >= 65:
        recommendation = "Good potential for renewable energy in South Africa."
    elif hybrid_score and hybrid_score >= 50:
        recommendation = "Moderate potential. Consider hybrid system."
    else:
        recommendation = "Limited potential. Consider other regions in SA."
    
    # Get 12-month forecast from ML
    forecast = ml_predictor.predict_monthly_forecast(lat, lon, 12)
    
    result = {
        'location': location_name,
        'lat': lat,
        'lon': lon,
        'solar_score': solar_score,
        'solar_irradiance': round(solar_irradiance, 1) if solar_irradiance else None,
        'wind_score': wind_score,
        'wind_speed': round(wind_speed, 1) if wind_speed else None,
        'hybrid_score': hybrid_score,
        'recommendation': recommendation,
        'data_source': data_source,
        'ml_scores': {
            'solar': ml_pred['solar'],
            'wind': ml_pred['wind'],
            'hybrid': ml_pred['hybrid'],
            'solar_confidence': ml_pred.get('solar_confidence', 85),
            'wind_confidence': ml_pred.get('wind_confidence', 85)
        },
        'forecast': forecast
    }
    
    # Cache the result
    cache.set(cache_key, result, timeout=3600)
    
    return jsonify(result)


@app.route('/api/forecast', methods=['POST'])
def get_forecast():
    """Get ML-based forecast for a location"""
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    months = data.get('months', 12)
    
    forecast = ml_predictor.predict_monthly_forecast(lat, lon, months)
    return jsonify({'forecast': forecast})


@app.route('/api/top', methods=['GET'])
def top_locations():
    """Return top SA locations (still uses predefined list but with live scores)"""
    # SA predefined locations
    locations = [
        {'name': 'Upington, Northern Cape', 'lat': -28.4, 'lon': 21.2},
        {'name': 'Springbok, Northern Cape', 'lat': -29.7, 'lon': 17.9},
        {'name': 'Cape Town, Western Cape', 'lat': -33.9, 'lon': 18.4},
        {'name': 'Jeffreys Bay, Eastern Cape', 'lat': -34.0, 'lon': 24.9},
        {'name': 'Richards Bay, KZN', 'lat': -28.8, 'lon': 32.1},
        {'name': 'Beaufort West, Western Cape', 'lat': -32.4, 'lon': 22.6},
        {'name': 'Kimberley, Northern Cape', 'lat': -28.7, 'lon': 24.8},
        {'name': 'Port Elizabeth, Eastern Cape', 'lat': -34.0, 'lon': 25.6},
        {'name': 'Mossel Bay, Western Cape', 'lat': -34.2, 'lon': 22.1},
        {'name': 'Saldanha, Western Cape', 'lat': -33.0, 'lon': 17.9},
    ]
    
    energy_type = request.args.get('type', 'hybrid')
    
    # Get ML scores for each location
    for loc in locations:
        pred = ml_predictor.predict_with_confidence(loc['lat'], loc['lon'])
        loc['solar'] = pred['solar']
        loc['wind'] = pred['wind']
        loc['hybrid'] = pred['hybrid']
    
    if energy_type == 'solar':
        locations.sort(key=lambda x: x['solar'], reverse=True)
    elif energy_type == 'wind':
        locations.sort(key=lambda x: x['wind'], reverse=True)
    else:
        locations.sort(key=lambda x: x['hybrid'], reverse=True)
    
    return jsonify({'energy_type': energy_type, 'count': len(locations), 'locations': locations[:10]})


@app.route('/api/heatmap', methods=['POST'])
def generate_heatmap():
    """Generate heatmap using ML predictions (NOT hardcoded estimates)"""
    data = request.get_json()
    bounds = data.get('bounds')
    energy_type = data.get('type', 'hybrid')
    resolution = data.get('resolution', 10)  # Higher resolution = slower
    
    lat_step = (bounds['north'] - bounds['south']) / resolution
    lon_step = (bounds['east'] - bounds['west']) / resolution
    
    heatmap_data = []
    
    # Cache grid predictions
    grid_cache = {}
    
    for i in range(resolution):
        lat = bounds['south'] + i * lat_step
        for j in range(resolution):
            lon = bounds['west'] + j * lon_step
            
            # Check cache
            cache_key = f"grid:{lat:.2f}:{lon:.2f}"
            if cache_key in grid_cache:
                score = grid_cache[cache_key]
            else:
                score = get_ml_score_for_grid(lat, lon, energy_type)
                grid_cache[cache_key] = score
            
            heatmap_data.append({
                'lat': round(lat, 4),
                'lon': round(lon, 4),
                'intensity': round(score / 100, 3)
            })
    
    return jsonify({'data': heatmap_data, 'resolution': resolution, 'points': len(heatmap_data)})


@app.route('/api/hotspots', methods=['POST'])
def find_hotspots():
    """Find high-potential clusters using ML predictions"""
    data = request.get_json()
    bounds = data.get('bounds')
    energy_type = data.get('type', 'hybrid')
    min_score = data.get('min_score', 70)
    
    # Scan grid
    resolution = 0.3  # ~33km grid
    lat_grid = np.arange(max(-35, bounds.get('min_lat', -35)), 
                         min(-22, bounds.get('max_lat', -22)), resolution)
    lon_grid = np.arange(max(16, bounds.get('min_lon', 16)), 
                         min(33, bounds.get('max_lon', 33)), resolution)
    
    high_potential = []
    
    print(f"Scanning {len(lat_grid) * len(lon_grid)} points for hotspots...")
    
    for lat in lat_grid:
        for lon in lon_grid:
            score = get_ml_score_for_grid(lat, lon, energy_type)
            if score >= min_score:
                high_potential.append({'lat': lat, 'lon': lon, 'score': score})
    
    # Cluster nearby points
    hotspots = []
    used = set()
    
    for i, point in enumerate(high_potential):
        if i in used:
            continue
        
        cluster = [point]
        cluster_indices = [i]
        
        for j, other in enumerate(high_potential):
            if j != i and j not in used:
                dist = math.sqrt((point['lat'] - other['lat'])**2 + (point['lon'] - other['lon'])**2)
                if dist < 1.5:  # ~165km radius
                    cluster.append(other)
                    cluster_indices.append(j)
        
        if len(cluster) >= 3:  # At least 3 points for a hotspot
            avg_lat = sum(p['lat'] for p in cluster) / len(cluster)
            avg_lon = sum(p['lon'] for p in cluster) / len(cluster)
            avg_score = sum(p['score'] for p in cluster) / len(cluster)
            radius_km = len(cluster) * 40
            
            hotspots.append({
                'center_lat': round(avg_lat, 2),
                'center_lon': round(avg_lon, 2),
                'radius_km': round(radius_km, 0),
                'intensity': round(avg_score, 1),
                'area_km2': round(3.14 * radius_km ** 2, 0),
                'point_count': len(cluster)
            })
            
            for idx in cluster_indices:
                used.add(idx)
    
    hotspots.sort(key=lambda x: x['intensity'], reverse=True)
    
    return jsonify({'hotspots': hotspots[:10], 'count': len(hotspots), 'total_points': len(high_potential)})


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'operational',
        'version': '3.0.0-south-africa-ml',
        'name': 'TerraVolt Intelligence - South Africa ML Edition',
        'bounds': SA_BOUNDS,
        'ml_models_loaded': ml_predictor.solar_model is not None and ml_predictor.wind_model is not None
    })


@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the API cache"""
    cache.clear()
    return jsonify({'status': 'cache_cleared'})


if __name__ == '__main__':
    print("=" * 60)
    print("🌍 TERRAVOLT INTELLIGENCE - SOUTH AFRICA ML EDITION")
    print("=" * 60)
    print("\n📍 Server running at: http://127.0.0.1:5000")
    print("📡 Focused on South African renewable energy potential")
    print(f"📍 Map bounds: {SA_BOUNDS}")
    print("\n📡 API endpoints:")
    print("   POST /api/analyze - Analyze SA location (live + ML)")
    print("   POST /api/forecast - Get 12-month ML forecast")
    print("   GET  /api/top     - Top SA locations (ML scores)")
    print("   POST /api/heatmap - Generate SA heatmap (ML-based)")
    print("   POST /api/hotspots - Find SA hotspots (ML-based)")
    print("   POST /api/clear_cache - Clear cache")
    print("\n✅ ML models loaded and ready")
    print("✅ Live API + ML fallback enabled")
    print("✅ Caching enabled (1 hour TTL)")
    print("\nPress Ctrl+C to stop\n")
    app.run(debug=True, port=5000)