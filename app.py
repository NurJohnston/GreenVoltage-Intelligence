"""
TERRAVOLT INTELLIGENCE - SOUTH AFRICA EDITION
AI-Powered Geospatial Renewable Energy Optimization Platform
Focused on South African renewable energy potential
"""

from flask import Flask, render_template, request, jsonify
import requests
import math
import numpy as np

app = Flask(__name__)

# South Africa bounds
SA_BOUNDS = {
    'min_lat': -35.0,
    'max_lat': -22.0,
    'min_lon': 16.0,
    'max_lon': 33.0
}

# South Africa specific locations
SA_LOCATIONS = [
    {'name': 'Upington, Northern Cape', 'lat': -28.4, 'lon': 21.2, 'solar': 95, 'wind': 65},
    {'name': 'Springbok, Northern Cape', 'lat': -29.7, 'lon': 17.9, 'solar': 94, 'wind': 70},
    {'name': 'Cape Town, Western Cape', 'lat': -33.9, 'lon': 18.4, 'solar': 85, 'wind': 88},
    {'name': 'Jeffreys Bay, Eastern Cape', 'lat': -34.0, 'lon': 24.9, 'solar': 82, 'wind': 92},
    {'name': 'Richards Bay, KZN', 'lat': -28.8, 'lon': 32.1, 'solar': 88, 'wind': 75},
    {'name': 'Beaufort West, Western Cape', 'lat': -32.4, 'lon': 22.6, 'solar': 92, 'wind': 80},
    {'name': 'Kimberley, Northern Cape', 'lat': -28.7, 'lon': 24.8, 'solar': 93, 'wind': 68},
    {'name': 'Port Elizabeth, Eastern Cape', 'lat': -34.0, 'lon': 25.6, 'solar': 84, 'wind': 86},
    {'name': 'Mossel Bay, Western Cape', 'lat': -34.2, 'lon': 22.1, 'solar': 86, 'wind': 85},
    {'name': 'Saldanha, Western Cape', 'lat': -33.0, 'lon': 17.9, 'solar': 87, 'wind': 89},
]

# ================================================================
# SOLAR SCORE CALCULATION (NASA POWER API - LIVE)
# ================================================================

def get_solar_score(lat, lon):
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
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        solar_data = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
        values = list(solar_data.values())
        
        if not values:
            return 50, 150
        
        avg_irradiance = sum(values) / len(values)
        score = min(100, max(0, (avg_irradiance - 50) / 250 * 100))
        return round(score, 1), avg_irradiance
        
    except Exception as e:
        print(f"Solar API error: {e}")
        return 50, 150


# ================================================================
# WIND SCORE CALCULATION (Open-Meteo API - LIVE)
# ================================================================

def get_wind_score(lat, lon):
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
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
            return 50, 5
        
        wind_speeds = data['hourly']['wind_speed_10m']
        valid_speeds = [s for s in wind_speeds if s is not None]
        
        if not valid_speeds:
            return 50, 5
        
        avg_wind_speed = sum(valid_speeds) / len(valid_speeds)
        score = min(100, max(0, (avg_wind_speed - 3) / 12 * 100))
        return round(score, 1), avg_wind_speed
        
    except Exception as e:
        print(f"Wind API error: {e}")
        return 50, 5


def get_hybrid_score(solar_score, wind_score):
    return round((solar_score + wind_score) / 2, 1)


def get_location_name(lat, lon):
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 10}
        headers = {'User-Agent': 'TerraVolt-Intelligence/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        if 'display_name' in data:
            parts = data['display_name'].split(',')
            if 'South Africa' in data['display_name']:
                return parts[0].strip() + ", South Africa"
            return parts[0].strip() if len(parts) >= 2 else parts[0].strip()
        return f"{lat:.2f}, {lon:.2f}"
    except:
        return f"{lat:.2f}, {lon:.2f}"


def get_top_locations(energy_type='hybrid', limit=10):
    locations = SA_LOCATIONS.copy()
    for loc in locations:
        loc['hybrid'] = round((loc['solar'] + loc['wind']) / 2, 1)
    
    if energy_type == 'solar':
        locations.sort(key=lambda x: x['solar'], reverse=True)
    elif energy_type == 'wind':
        locations.sort(key=lambda x: x['wind'], reverse=True)
    else:
        locations.sort(key=lambda x: x['hybrid'], reverse=True)
    
    return locations[:limit]


# ================================================================
# SOUTH AFRICA HEATMAP ESTIMATION
# ================================================================

def estimate_solar_score_sa(lat, lon):
    """Solar estimation specifically for South Africa"""
    abs_lat = abs(lat)
    # Northern Cape has best solar (around -28 to -30)
    if -30 <= lat <= -28:
        base = 95
    elif lat < -30:
        base = 85 - (abs_lat - 30) * 2
    else:
        base = 85 - (abs_lat - 28) * 1.5
    return max(60, min(98, base))


def estimate_wind_score_sa(lat, lon):
    """Wind estimation specifically for South Africa"""
    # Coastal areas have best wind
    # Check if near coast (simplified)
    is_coastal = (lon < 18.5 and lat > -34.5) or (lon > 25 and lat > -34) or (lat < -34)
    
    if is_coastal:
        if -35 <= lat <= -33:
            return 88 + (34 - abs(lat)) * 2
        return 85
    else:
        # Inland areas
        if -30 <= lat <= -28:
            return 68
        return 65


# ================================================================
# FLASK ROUTES
# ================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    lat = data.get('lat')
    lon = data.get('lon')
    
    if lat is None or lon is None:
        return jsonify({'error': 'Missing coordinates'}), 400
    
    location_name = get_location_name(lat, lon)
    solar_score, solar_irradiance = get_solar_score(lat, lon)
    wind_score, wind_speed = get_wind_score(lat, lon)
    hybrid_score = get_hybrid_score(solar_score, wind_score)
    
    if hybrid_score >= 80:
        recommendation = "Excellent location for renewable energy in South Africa!"
    elif hybrid_score >= 65:
        recommendation = "Good potential for renewable energy in South Africa."
    elif hybrid_score >= 50:
        recommendation = "Moderate potential. Consider hybrid system."
    else:
        recommendation = "Limited potential. Consider other regions in SA."
    
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
    energy_type = request.args.get('type', 'hybrid')
    limit = int(request.args.get('limit', 10))
    locations = get_top_locations(energy_type, limit)
    return jsonify({'energy_type': energy_type, 'count': len(locations), 'locations': locations})


@app.route('/api/heatmap', methods=['POST'])
def generate_heatmap():
    data = request.get_json()
    bounds = data.get('bounds')
    energy_type = data.get('type', 'hybrid')
    resolution = data.get('resolution', 30)
    
    lat_step = (bounds['north'] - bounds['south']) / resolution
    lon_step = (bounds['east'] - bounds['west']) / resolution
    
    heatmap_data = []
    
    for i in range(resolution):
        lat = bounds['south'] + i * lat_step
        for j in range(resolution):
            lon = bounds['west'] + j * lon_step
            
            if energy_type == 'solar':
                score = estimate_solar_score_sa(lat, lon)
            elif energy_type == 'wind':
                score = estimate_wind_score_sa(lat, lon)
            else:
                score = (estimate_solar_score_sa(lat, lon) + estimate_wind_score_sa(lat, lon)) / 2
            
            heatmap_data.append({
                'lat': round(lat, 4),
                'lon': round(lon, 4),
                'intensity': round(score / 100, 3)
            })
    
    return jsonify({'data': heatmap_data})


@app.route('/api/hotspots', methods=['POST'])
def find_hotspots():
    data = request.get_json()
    bounds = data.get('bounds')
    energy_type = data.get('type', 'hybrid')
    min_score = data.get('min_score', 70)
    
    # Scan SA grid
    resolution = 0.5
    lat_grid = np.arange(max(-35, bounds['min_lat']), min(-22, bounds['max_lat']), resolution)
    lon_grid = np.arange(max(16, bounds['min_lon']), min(33, bounds['max_lon']), resolution)
    
    high_potential = []
    
    for lat in lat_grid:
        for lon in lon_grid:
            if energy_type == 'solar':
                score = estimate_solar_score_sa(lat, lon)
            elif energy_type == 'wind':
                score = estimate_wind_score_sa(lat, lon)
            else:
                score = (estimate_solar_score_sa(lat, lon) + estimate_wind_score_sa(lat, lon)) / 2
            
            if score >= min_score:
                high_potential.append({'lat': lat, 'lon': lon, 'score': score})
    
    # Group nearby points into clusters
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
                if dist < 1.5:
                    cluster.append(other)
                    cluster_indices.append(j)
        
        if len(cluster) >= 2:
            avg_lat = sum(p['lat'] for p in cluster) / len(cluster)
            avg_lon = sum(p['lon'] for p in cluster) / len(cluster)
            avg_score = sum(p['score'] for p in cluster) / len(cluster)
            radius_km = len(cluster) * 55
            
            hotspots.append({
                'center_lat': round(avg_lat, 2),
                'center_lon': round(avg_lon, 2),
                'radius_km': round(radius_km, 0),
                'intensity': round(avg_score, 1),
                'area_km2': round(3.14 * radius_km ** 2, 0)
            })
            
            for idx in cluster_indices:
                used.add(idx)
    
    hotspots.sort(key=lambda x: x['intensity'], reverse=True)
    
    return jsonify({'hotspots': hotspots[:10], 'count': len(hotspots)})


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'operational',
        'version': '2.0.0-south-africa',
        'name': 'TerraVolt Intelligence - South Africa Edition',
        'bounds': SA_BOUNDS
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🌍 TERRAVOLT INTELLIGENCE - SOUTH AFRICA EDITION")
    print("=" * 60)
    print("\n📍 Server running at: http://127.0.0.1:5000")
    print("📡 Focused on South African renewable energy potential")
    print(f"📍 Map bounds: {SA_BOUNDS}")
    print("\n📡 API endpoints:")
    print("   POST /api/analyze - Analyze SA location")
    print("   GET  /api/top     - Top SA locations")
    print("   POST /api/heatmap - Generate SA heatmap")
    print("   POST /api/hotspots - Find SA hotspots")
    print("\n✅ Using live API data for point analysis")
    print("✅ Using SA-specific estimation for heatmap")
    print("\nPress Ctrl+C to stop\n")
    app.run(debug=True, port=5000)