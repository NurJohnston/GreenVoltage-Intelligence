import requests
from datetime import datetime, timedelta
import time
from database import store_weather_data, init_database, SolarHistorical, WindHistorical
import numpy as np
from tqdm import tqdm
import calendar

class HistoricalDataCollector:
    def __init__(self):
        self.session, self.engine = init_database()
    
    def collect_sa_locations(self):
        """Collect data for key South African locations"""
        sa_locations = [
            # Northern Cape (best solar)
            {'name': 'Upington', 'lat': -28.4, 'lon': 21.2},
            {'name': 'Springbok', 'lat': -29.7, 'lon': 17.9},
            {'name': 'Kimberley', 'lat': -28.7, 'lon': 24.8},
            {'name': 'Carnarvon', 'lat': -30.0, 'lon': 22.1},
            {'name': 'Pofadder', 'lat': -29.1, 'lon': 19.4},
            
            # Western Cape (best wind)
            {'name': 'Cape Town', 'lat': -33.9, 'lon': 18.4},
            {'name': 'Mossel Bay', 'lat': -34.2, 'lon': 22.1},
            {'name': 'Saldanha', 'lat': -33.0, 'lon': 17.9},
            {'name': 'Beaufort West', 'lat': -32.4, 'lon': 22.6},
            {'name': 'Hermanus', 'lat': -34.4, 'lon': 19.2},
            
            # Eastern Cape
            {'name': 'Jeffreys Bay', 'lat': -34.0, 'lon': 24.9},
            {'name': 'Port Elizabeth', 'lat': -34.0, 'lon': 25.6},
            {'name': 'East London', 'lat': -33.0, 'lon': 27.9},
            
            # KZN
            {'name': 'Richards Bay', 'lat': -28.8, 'lon': 32.1},
            {'name': 'Durban', 'lat': -29.9, 'lon': 31.0},
            
            # Inland
            {'name': 'Johannesburg', 'lat': -26.2, 'lon': 28.0},
            {'name': 'Pretoria', 'lat': -25.7, 'lon': 28.2},
            {'name': 'Bloemfontein', 'lat': -29.1, 'lon': 26.2},
            {'name': 'Polokwane', 'lat': -23.9, 'lon': 29.5},
        ]
        return sa_locations
    
    def get_current_date(self):
        """Get the most recent complete month"""
        now = datetime.now()
        # Go back to first day of current month to ensure data is available
        return datetime(now.year, now.month, 1) - timedelta(days=1)
    
    def collect_decades_of_data(self, lat, lon, start_year=1950, end_year=None):
        """Collect decades of monthly data for a location up to current date"""
        if end_year is None:
            end_year = datetime.now().year
        
        print(f"\n📡 Collecting {end_year - start_year + 1} years of data for ({lat}, {lon})...")
        
        monthly_data = []
        current_date = self.get_current_date()
        
        # Process in batches to avoid overwhelming APIs
        for year in tqdm(range(start_year, end_year + 1), desc=f"Years", leave=False):
            max_month = 12
            if year == end_year:
                max_month = current_date.month
            
            for month in range(1, max_month + 1):
                date = datetime(year, month, 15)
                
                # Skip future dates
                if date > current_date:
                    continue
                
                # Skip if data already exists
                existing = self.session.query(SolarHistorical).filter(
                    SolarHistorical.lat == lat,
                    SolarHistorical.lon == lon,
                    SolarHistorical.date == date
                ).first()
                
                if existing:
                    continue
                
                # Get solar data for this month
                solar_data = self._get_monthly_solar(lat, lon, year, month)
                
                # Get wind data for this month
                wind_data = self._get_monthly_wind(lat, lon, year, month)
                
                if solar_data or wind_data:
                    store_weather_data(
                        self.session, lat, lon, date,
                        solar_data=solar_data,
                        wind_data=wind_data
                    )
                    monthly_data.append({'date': date, 'solar': solar_data, 'wind': wind_data})
                
                # Rate limiting - be respectful to APIs
                time.sleep(0.3)
            
            # Commit after each year
            self.session.commit()
        
        print(f"  ✅ Collected {len(monthly_data)} months of data")
        return monthly_data
    
    def _get_monthly_solar(self, lat, lon, year, month):
        """Get monthly average solar irradiance from NASA POWER"""
        try:
            # Get daily data for the month
            start_date = f"{year}{month:02d}01"
            
            # Calculate end date (first day of next month)
            if month == 12:
                if year >= datetime.now().year:
                    # Don't go beyond current date
                    end_date = datetime.now().strftime("%Y%m%d")
                else:
                    end_date = f"{year+1}0101"
            else:
                end_date = f"{year}{month+1:02d}01"
            
            url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            params = {
                'parameters': 'ALLSKY_SFC_SW_DWN',
                'community': 'RE',
                'longitude': lon,
                'latitude': lat,
                'start': start_date,
                'end': end_date,
                'format': 'JSON'
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'properties' in data and 'parameter' in data['properties']:
                solar_data = data['properties']['parameter'].get('ALLSKY_SFC_SW_DWN', {})
                values = [v for v in solar_data.values() if v is not None]
                
                if values:
                    avg_irradiance = sum(values) / len(values)
                    
                    # Estimate cloud cover from irradiance
                    cloud_cover = max(0, min(100, 100 - (avg_irradiance / 350 * 100)))
                    
                    return {
                        'irradiance': avg_irradiance,
                        'uv_index': avg_irradiance / 50,
                        'cloud_cover': cloud_cover
                    }
            
            return None
            
        except Exception as e:
            print(f"    Solar API error for {year}-{month}: {e}")
            return None
    
    def _get_monthly_wind(self, lat, lon, year, month):
        """Get monthly average wind data from Open-Meteo"""
        try:
            # Open-Meteo has data from 1940 onwards, including recent months
            start_date = f"{year}-{month:02d}-01"
            
            if month == 12:
                if year >= datetime.now().year:
                    end_date = datetime.now().strftime("%Y-%m-%d")
                else:
                    end_date = f"{year+1}-01-01"
            else:
                end_date = f"{year}-{month+1:02d}-01"
            
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                'latitude': lat,
                'longitude': lon,
                'start_date': start_date,
                'end_date': end_date,
                'daily': 'wind_speed_10m_max',
                'timezone': 'auto'
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            if 'daily' in data and 'wind_speed_10m_max' in data['daily']:
                wind_speeds = [s for s in data['daily']['wind_speed_10m_max'] if s is not None]
                
                if wind_speeds:
                    avg_wind_speed = sum(wind_speeds) / len(wind_speeds)
                    
                    # Calculate wind power density
                    air_density = 1.225
                    wind_power = 0.5 * air_density * (avg_wind_speed ** 3)
                    
                    return {
                        'wind_speed': avg_wind_speed,
                        'wind_direction': 0,
                        'wind_gust': avg_wind_speed * 1.5,
                        'wind_power_density': wind_power
                    }
            
            return None
            
        except Exception as e:
            print(f"    Wind API error for {year}-{month}: {e}")
            return None
    
    def collect_all_sa_data(self, start_year=1950, end_year=None, max_locations=None):
        """Collect data for all major SA locations up to current date"""
        if end_year is None:
            end_year = datetime.now().year
        
        locations = self.collect_sa_locations()
        
        if max_locations:
            locations = locations[:max_locations]
        
        total_years = end_year - start_year + 1
        total_months = len(locations) * total_years * 12
        
        current_date = datetime.now()
        
        print("=" * 60)
        print("🌍 Collecting Decades of Historical Data for South Africa")
        print("=" * 60)
        print(f"📍 Locations: {len(locations)}")
        print(f"📅 Date range: {start_year} to {current_date.strftime('%B %Y')}")
        print(f"📊 Total years: {total_years}")
        print(f"📊 Total possible samples: {total_months}")
        print(f"⏱️ Estimated time: ~{total_months * 0.3 / 60:.1f} hours")
        print("=" * 60)
        
        # Show what we'll collect
        print(f"\n📡 Data sources:")
        print("   ☀️ NASA POWER: 1984 - present (solar irradiance)")
        print("   💨 Open-Meteo: 1940 - present (wind speed)")
        print("   📅 Latest data: Up to last complete month")
        print("=" * 60)
        
        confirm = input("\nThis will take several hours. Continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return
        
        for location in tqdm(locations, desc="Locations"):
            print(f"\n📍 {location['name']} ({location['lat']}, {location['lon']})")
            self.collect_decades_of_data(
                location['lat'], 
                location['lon'], 
                start_year, 
                end_year
            )
            # Commit after each location
            self.session.commit()
        
        print("\n" + "=" * 60)
        print("✅ Data collection complete!")
        self.verify_data()
        self.check_data_quality()
        print("=" * 60)
    
    def verify_data(self):
        """Check how much data was collected"""
        from sqlalchemy import func
        
        solar_count = self.session.query(func.count(SolarHistorical.id)).scalar()
        wind_count = self.session.query(func.count(WindHistorical.id)).scalar()
        
        print(f"\n📊 Database Statistics:")
        print(f"   Solar records: {solar_count}")
        print(f"   Wind records: {wind_count}")
        
        if solar_count > 0:
            # Show date range
            oldest = self.session.query(SolarHistorical).order_by(SolarHistorical.date).first()
            newest = self.session.query(SolarHistorical).order_by(SolarHistorical.date.desc()).first()
            if oldest and newest:
                print(f"   Date range: {oldest.date.strftime('%Y-%m-%d')} to {newest.date.strftime('%Y-%m-%d')}")
            
            # Show sample
            sample = self.session.query(SolarHistorical).first()
            if sample:
                print(f"   Sample: {sample.lat}, {sample.lon}, {sample.date}, {sample.irradiance:.1f} W/m²")
        
        return solar_count, wind_count
    
    def check_data_quality(self):
        """Check if data looks reasonable for South Africa"""
        from sqlalchemy import func
        
        # SA typical ranges
        # Solar: 50-350 W/m² (higher in Northern Cape)
        bad_solar = self.session.query(SolarHistorical).filter(
            (SolarHistorical.irradiance < 50) | (SolarHistorical.irradiance > 400)
        ).count()
        
        # Wind: 0-25 m/s (coastal areas can be higher)
        bad_wind = self.session.query(WindHistorical).filter(
            (WindHistorical.wind_speed < 0) | (WindHistorical.wind_speed > 35)
        ).count()
        
        print(f"\n🔍 Data Quality Check:")
        print(f"   Suspicious solar records (<50 or >400 W/m²): {bad_solar}")
        print(f"   Suspicious wind records (<0 or >35 m/s): {bad_wind}")
        
        # Show average values for last year
        one_year_ago = datetime.now() - timedelta(days=365)
        recent_solar = self.session.query(func.avg(SolarHistorical.irradiance)).filter(
            SolarHistorical.date >= one_year_ago
        ).scalar()
        
        recent_wind = self.session.query(func.avg(WindHistorical.wind_speed)).filter(
            WindHistorical.date >= one_year_ago
        ).scalar()
        
        if recent_solar and recent_wind:
            print(f"\n   📈 Recent 12-month averages:")
            print(f"      Solar irradiance: {recent_solar:.1f} W/m²")
            print(f"      Wind speed: {recent_wind:.1f} m/s")
        
        if bad_solar == 0 and bad_wind == 0:
            print("\n   ✅ Data looks reasonable for South Africa!")
        else:
            print("\n   ⚠️ Some unusual values found - may need cleaning")


if __name__ == '__main__':
    print("=" * 60)
    print("🌍 TERRAVOLT HISTORICAL DATA COLLECTOR - FULL HISTORICAL")
    print("=" * 60)
    
    collector = HistoricalDataCollector()
    
    # Check current data
    solar_count, wind_count = collector.verify_data()
    
    current_date = datetime.now()
    
    if solar_count == 0:
        print("\n📦 No data found. Starting fresh collection...")
        
        print("\nCollection Options:")
        print("1. Quick test (2020-2024, 5 locations) - ~15 minutes")
        print("2. Full SA (1984-2026, 20 locations) - ~6 hours")
        print("3. Maximum (1940-2026, 20 locations) - ~10 hours")
        print(f"4. Custom date range")
        
        choice = input("\nChoose option (1/2/3/4): ")
        
        if choice == '1':
            collector.collect_all_sa_data(start_year=2020, end_year=current_date.year, max_locations=5)
        elif choice == '2':
            # NASA POWER data is reliable from 1984 onwards
            collector.collect_all_sa_data(start_year=1984, end_year=current_date.year)
        elif choice == '3':
            # Open-Meteo has data from 1940, NASA from 1984
            print("\n⚠️ Note: Solar data only available from 1984 onwards")
            collector.collect_all_sa_data(start_year=1940, end_year=current_date.year)
        elif choice == '4':
            start = int(input("Start year (e.g., 1950): "))
            end = int(input(f"End year (max {current_date.year}): "))
            collector.collect_all_sa_data(start_year=start, end_year=end)
        else:
            print("Invalid choice. Cancelled.")
    else:
        print(f"\n✅ Found {solar_count} existing records")
        collector.check_data_quality()
        
        response = input("\nAdd more historical data (including recent months)? (y/n): ")
        if response.lower() == 'y':
            start_year = int(input("Start year (e.g., 1950): "))
            end_year = current_date.year
            collector.collect_all_sa_data(start_year=start_year, end_year=end_year)