import requests
from datetime import datetime, timedelta
import numpy as np

class LiveDataProvider:
    def __init__(self):
        pass
    
    def get_live_solar(self, lat, lon):
        """Get real-time solar irradiance and forecast"""
        # Using Open-Meteo's real-time API
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'shortwave_radiation',
            'hourly': 'shortwave_radiation',
            'forecast_days': 7,
            'timezone': 'auto'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            if 'current' in data:
                current_irradiance = data['current']['shortwave_radiation']
                
                # Get 7-day forecast
                forecast = []
                for i, rad in enumerate(data['hourly']['shortwave_radiation'][:168]):  # 7 days * 24h
                    forecast.append({
                        'hour': i,
                        'irradiance': rad
                    })
                
                return {
                    'current_irradiance': current_irradiance,
                    'forecast': forecast,
                    'source': 'live'
                }
        except Exception as e:
            print(f"Live solar error: {e}")
        
        return None
    
    def get_live_wind(self, lat, lon):
        """Get real-time wind speed, direction, and gusts"""
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': 'wind_speed_10m,wind_direction_10m,wind_gusts_10m',
            'hourly': 'wind_speed_10m,wind_direction_10m',
            'forecast_days': 7,
            'timezone': 'auto'
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            if 'current' in data:
                current = data['current']
                
                # Calculate wind power density (W/m²)
                air_density = 1.225
                wind_power = 0.5 * air_density * (current['wind_speed_10m'] ** 3)
                
                # 7-day forecast
                forecast = []
                for i in range(168):
                    forecast.append({
                        'hour': i,
                        'wind_speed': data['hourly']['wind_speed_10m'][i],
                        'wind_direction': data['hourly']['wind_direction_10m'][i]
                    })
                
                return {
                    'current_speed': current['wind_speed_10m'],
                    'current_direction': current['wind_direction_10m'],
                    'current_gust': current.get('wind_gusts_10m', 0),
                    'current_power_wm2': wind_power,
                    'forecast': forecast,
                    'source': 'live'
                }
        except Exception as e:
            print(f"Live wind error: {e}")
        
        return None
    
    def get_hybrid_live(self, lat, lon):
        """Combine live solar and wind data"""
        solar = self.get_live_solar(lat, lon)
        wind = self.get_live_wind(lat, lon)
        
        result = {'timestamp': datetime.now().isoformat()}
        
        if solar:
            result['solar'] = solar
            # Convert to score
            solar_score = min(100, max(0, (solar['current_irradiance'] - 50) / 250 * 100))
            result['solar_score'] = round(solar_score, 1)
        
        if wind:
            result['wind'] = wind
            # Convert to score
            wind_score = min(100, max(0, (wind['current_speed'] - 3) / 12 * 100))
            result['wind_score'] = round(wind_score, 1)
        
        if solar and wind:
            result['hybrid_score'] = round((result['solar_score'] + result['wind_score']) / 2, 1)
        
        return result