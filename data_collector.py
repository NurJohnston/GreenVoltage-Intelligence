import requests
from datetime import datetime
import time
from database import store_weather_data, init_database
import numpy as np

class HistoricalDataCollector:
    def __init__(self):
        self.session, self.engine = init_database()
    
    def collect_location_history(self, lat, lon, start_year=2010, end_year=2023):
        """Collect decades of historical data for a location"""
        print(f"Collecting data for {lat}, {lon} from {start_year} to {end_year}")
        
        for year in range(start_year, end_year + 1):
            print(f"  Processing year {year}...")
            
            # Solar data from NASA POWER
            solar_data = self._fetch_nasa_power_year(lat, lon, year)
            
            # Wind data from Open-Meteo
            wind_data = self._fetch_openmeteo_year(lat, lon, year)
            
            # Store monthly averages
            for month in range(1, 13):
                date = datetime(year, month, 15)
                
                store_weather_data(
                    self.session, lat, lon, date,
                    solar_data=solar_data.get(month) if solar_data else None,
                    wind_data=wind_data.get(month) if wind_data else None
                )
            
            # Be respectful of API limits
            time.sleep(1)
        
        print(f"Completed data collection for {lat}, {lon}")
    
    def _fetch_nasa_power_year(self, lat, lon, year):
        """Fetch one year of NASA POWER data - FIXED version"""
        url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            'parameters': 'ALLSKY_SFC_SW_DWN',
            'community': 'RE',
            'longitude': lon,
            'latitude': lat,
            'start': f'{year}0101',
            'end': f'{year}1231',
            'format': 'JSON'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # Check if we have the expected data structure
            if 'properties' not in data:
                print(f"    NASA POWER: No properties in response for {year}")
                return {}
            
            if 'parameter' not in data['properties']:
                print(f"    NASA POWER: No parameter in properties for {year}")
                return {}
            
            if 'ALLSKY_SFC_SW_DWN' not in data['properties']['parameter']:
                print(f"    NASA POWER: No ALLSKY_SFC_SW_DWN data for {year}")
                return {}
            
            solar_params = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
            
            # Organize data by month
            monthly_data = {month: [] for month in range(1, 13)}
            
            for date_str, irradiance in solar_params.items():
                if irradance is not None:
                    # date_str format: YYYYMMDD
                    month = int(date_str[4:6])
                    monthly_data[month].append(irradiance)
            
            # Calculate monthly averages
            result = {}
            for month in range(1, 13):
                if monthly_data[month]:
                    avg_irradiance = sum(monthly_data[month]) / len(monthly_data[month])
                    result[month] = {
                        'irradiance': avg_irradiance,
                        'cloud_cover': max(0, min(100, 100 - (avg_irradiance / 350 * 100))),
                        'uv_index': avg_irradiance / 50
                    }
                else:
                    result[month] = None
            
            # Count how many months have data
            months_with_data = sum(1 for m in result.values() if m is not None)
            print(f"    NASA POWER: Retrieved {months_with_data}/12 months for {year}")
            return result
            
        except Exception as e:
            print(f"    NASA POWER error for {year}: {e}")
            return {}
    
    def _fetch_openmeteo_year(self, lat, lon, year):
        """Fetch one year of wind data from Open-Meteo"""
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': f'{year}-01-01',
            'end_date': f'{year}-12-31',
            'daily': 'wind_speed_10m_max,wind_direction_10m_dominant',
            'timezone': 'auto'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'daily' not in data or 'time' not in data['daily']:
                print(f"    Open-Meteo: No wind data for {year}")
                return {}
            
            monthly_data = {month: [] for month in range(1, 13)}
            
            # Group by month
            for i, date_str in enumerate(data['daily']['time']):
                date = datetime.strptime(date_str, '%Y-%m-%d')
                month = date.month
                
                wind_speed = data['daily']['wind_speed_10m_max'][i]
                wind_direction = data['daily'].get('wind_direction_10m_dominant', [0] * len(data['daily']['time']))[i]
                
                if wind_speed is not None:
                    monthly_data[month].append({
                        'speed': wind_speed,
                        'direction': wind_direction if wind_direction else 0
                    })
            
            # Calculate monthly averages
            result = {}
            for month in range(1, 13):
                if monthly_data[month]:
                    avg_speed = np.mean([v['speed'] for v in monthly_data[month]])
                    avg_direction = np.mean([v['direction'] for v in monthly_data[month]])
                    result[month] = {
                        'wind_speed': avg_speed,
                        'wind_direction': avg_direction,
                        'wind_gust': avg_speed * 1.5  # Estimate gust
                    }
                else:
                    result[month] = None
            
            months_with_data = sum(1 for m in result.values() if m is not None)
            print(f"    Open-Meteo: Retrieved {months_with_data}/12 months for {year}")
            return result
            
        except Exception as e:
            print(f"    Open-Meteo error for {year}: {e}")
            return {}
    
    def collect_global_grid(self, resolution=5):
        """Collect data for global grid"""
        latitudes = np.arange(-90, 91, resolution)
        longitudes = np.arange(-180, 181, resolution)
        
        total_points = len(latitudes) * len(longitudes)
        print(f"Collecting data for {total_points} global grid points")
        
        for lat in latitudes:
            for lon in longitudes:
                self.collect_location_history(lat, lon, start_year=2020, end_year=2023)
        
        print("Global data collection complete!")


if __name__ == '__main__':
    print("=" * 60)
    print("🌍 TERRAVOLT DATA COLLECTOR")
    print("=" * 60)
    
    collector = HistoricalDataCollector()
    
    # Start with known hot spots
    hotspots = [
        (23.0, 13.0),   # Sahara Desert
        (-24.5, -69.0), # Atacama Desert
        (55.0, 3.0),    # North Sea
        (-34.0, 22.0),  # South Africa Coast
        (40.0, -100.0), # Great Plains
        (27.0, 71.0),   # Thar Desert
        (42.0, 105.0),  # Gobi Desert
    ]
    
    print("\n📊 Starting data collection for hotspots...")
    print("This will take a few minutes per location...\n")
    
    for lat, lon in hotspots:
        collector.collect_location_history(lat, lon, start_year=2020, end_year=2023)
    
    print("\n" + "=" * 60)
    print("✅ Data collection complete!")
    print("=" * 60)