import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from database import init_database, SolarHistorical, WindHistorical
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import os

class RenewableEnergyPredictor:
    def __init__(self):
        self.session, self.engine = init_database()
        self.solar_model = None
        self.wind_model = None
        self.scaler = StandardScaler()
    
    def load_models(self):
        """Load trained models from disk"""
        #solar_path = Path('solar_model.pkl')
        solar_path = Path(__file__).parent / 'solar_model.pkl'
        #wind_path = Path('wind_model.pkl')
        wind_path = Path(__file__).parent / 'wind_model.pkl'
        
        if solar_path.exists():
            self.solar_model = joblib.load(solar_path)
            print("✅ Solar model loaded")
        else:
            print("⚠️ Solar model not found, will train on next run")
        
        if wind_path.exists():
            self.wind_model = joblib.load(wind_path)
            print("✅ Wind model loaded")
        else:
            print("⚠️ Wind model not found, will train on next run")
    
    def prepare_training_data(self, years_back=10):
        """Extract historical data for ML training"""
        query_date = datetime.now() - timedelta(days=years_back*365)
        
        # Query solar data
        solar_query = self.session.query(SolarHistorical).filter(
            SolarHistorical.date >= query_date
        )
        solar_df = pd.read_sql(solar_query.statement, self.engine)
        
        # Query wind data
        wind_query = self.session.query(WindHistorical).filter(
            WindHistorical.date >= query_date
        )
        wind_df = pd.read_sql(wind_query.statement, self.engine)
        
        # Feature engineering
        solar_features = self._engineer_features(solar_df, 'solar')
        wind_features = self._engineer_features(wind_df, 'wind')
        
        return solar_features, wind_features
    
    def _engineer_features(self, df, energy_type):
        """Create ML features from raw data"""
        if df.empty:
            return None
        
        # Convert date to features
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
        df['season'] = df['month'] % 12 // 3 + 1
        
        # Location features (sin/cos for cyclical coordinates)
        df['lat_rad'] = np.radians(df['lat'])
        df['lon_rad'] = np.radians(df['lon'])
        df['lat_sin'] = np.sin(df['lat_rad'])
        df['lat_cos'] = np.cos(df['lat_rad'])
        df['lon_sin'] = np.sin(df['lon_rad'])
        df['lon_cos'] = np.cos(df['lon_rad'])
        
        # Temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        if energy_type == 'solar':
            # Target: irradiance
            features = ['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 
                       'month_sin', 'month_cos', 'day_of_year']
            X = df[features]
            y = df['irradiance']
        else:
            # Target: wind_power_density
            features = ['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos',
                       'month_sin', 'month_cos', 'day_of_year']
            X = df[features]
            y = df['wind_power_density']
        
        return X, y
    
    def train_models(self):
        """Train Random Forest models for solar and wind prediction"""
        print("Preparing training data...")
        solar_data, wind_data = self.prepare_training_data(years_back=5)
        
        if solar_data and solar_data[0] is not None:
            X_solar, y_solar = solar_data
            print(f"Training solar model on {len(X_solar)} samples...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_solar, y_solar, test_size=0.2, random_state=42
            )
            
            # Train Random Forest
            self.solar_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            self.solar_model.fit(X_train, y_train)
            
            # Evaluate
            score = self.solar_model.score(X_test, y_test)
            print(f"Solar model R² score: {score:.3f}")
            
            # Cross-validation
            cv_scores = cross_val_score(self.solar_model, X_solar, y_solar, cv=5)
            print(f"Solar CV scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            # Save model
            joblib.dump(self.solar_model, 'solar_model.pkl')
            print("✅ Solar model saved to solar_model.pkl")
        else:
            print("⚠️ No solar training data available")
        
        if wind_data and wind_data[0] is not None:
            X_wind, y_wind = wind_data
            print(f"Training wind model on {len(X_wind)} samples...")
            
            # Use Gradient Boosting for wind (often better for non-linear patterns)
            self.wind_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.wind_model.fit(X_wind, y_wind)
            
            score = self.wind_model.score(X_wind, y_wind)
            print(f"Wind model R² score: {score:.3f}")
            
            # Save model
            joblib.dump(self.wind_model, 'wind_model.pkl')
            print("✅ Wind model saved to wind_model.pkl")
        else:
            print("⚠️ No wind training data available")
        
        print("Model training complete!")
    
    def predict_location(self, lat, lon, date=None):
        """Predict renewable energy potential for a location"""
        if date is None:
            date = datetime.now()
        
        # Create feature vector
        features = self._create_prediction_features(lat, lon, date)
        features_array = np.array([features])
        
        # Make predictions
        solar_pred = 0
        wind_pred = 0
        
        if self.solar_model:
            solar_pred = self.solar_model.predict(features_array)[0]
        
        if self.wind_model:
            wind_pred = self.wind_model.predict(features_array)[0]
        
        # Convert to scores (0-100)
        #solar_score = min(100, max(0, (solar_pred - 50) / 250 * 100))
        solar_score = min(100, max(0, (solar_pred - 1.46) / 8.11 * 100))
        wind_score = min(100, max(0, (wind_pred - 10) / 500 * 100))  # Wind power density scaling
        
        return {
            'solar': round(solar_score, 1),
            'wind': round(wind_score, 1),
            'hybrid': round((solar_score + wind_score) / 2, 1),
            'solar_power_wm2': round(solar_pred, 1),
            'wind_power_wm2': round(wind_pred, 1)
        }
    
    def predict_with_confidence(self, lat, lon, date=None):
        """Predict with confidence interval"""
        if date is None:
            date = datetime.now()
        
        features = self._create_prediction_features(lat, lon, date)
        #features_array = np.array([features])
        features_array = features
        
        # Get predictions with confidence (using tree variance for RF)
        solar_pred = 0
        wind_pred = 0
        solar_std = 0
        wind_std = 0
        
        if self.solar_model:
            if hasattr(self.solar_model, 'estimators_'):
                # Random Forest - get variance across trees
                predictions = [tree.predict(features_array)[0] for tree in self.solar_model.estimators_]
                solar_pred = np.mean(predictions)
                solar_std = np.std(predictions)
            else:
                solar_pred = self.solar_model.predict(features_array)[0]
                solar_std = solar_pred * 0.1  # Assume 10% uncertainty
        
        if self.wind_model:
            #if hasattr(self.wind_model, 'estimators_'):
            #    predictions = [tree.predict(features_array)[0] for tree in self.wind_model.estimators_]
            if hasattr(self.wind_model, 'estimators_') and hasattr(self.wind_model.estimators_[0], 'predict'):
                wind_pred = np.mean(predictions)
                wind_std = np.std(predictions)
            else:
                wind_pred = self.wind_model.predict(features_array)[0]
                wind_std = wind_pred * 0.15
        
        # Convert to scores
        #solar_score = min(100, max(0, (solar_pred - 50) / 250 * 100))
        #solar_score = min(100, max(0, (solar_pred - 100) / 900 * 100))
        #solar_score = min(100, max(0, (solar_pred - 289) / 988 * 100))
        #solar_score = min(100, max(0, (solar_pred - 200) / 800 * 100))
        solar_score = min(100, max(0, (solar_pred - 1.46) / 8.11 * 100))
        #wind_score = min(100, max(0, (wind_pred - 10) / 500 * 100))
        #wind_score = min(100, max(0, (wind_pred - 10) / 800 * 100))
        #wind_score = min(100, max(0, (wind_pred - 14) / (2028 - 14) * 100))
        wind_score = min(100, max(0, (wind_pred - 50) / 500 * 100))
        
        return {
            'solar': round(solar_score, 1),
            'wind': round(wind_score, 1),
            'hybrid': round((solar_score + wind_score) / 2, 1),
            'solar_power_wm2': round(solar_pred, 1),
            'wind_power_wm2': round(wind_pred, 1),
            'solar_confidence': round(max(0, min(100, 100 - (solar_std / max(solar_pred, 1) * 100))), 1),
            'wind_confidence': round(max(0, min(100, 100 - (wind_std / max(wind_pred, 1) * 100))), 1)
        }
    
    def predict_monthly_forecast(self, lat, lon, months=12):
        """Generate monthly forecast for a location"""
        forecasts = []
        base_date = datetime.now().replace(day=15)
        
        for i in range(months):
            forecast_date = base_date + timedelta(days=30 * i)
            pred = self.predict_with_confidence(lat, lon, forecast_date)
            forecasts.append({
                'month': forecast_date.strftime('%b %Y'),
                'solar': pred['solar'],
                'wind': pred['wind'],
                'hybrid': pred['hybrid']
            })
        
        return forecasts

    def predict_annual_average(self, lat, lon):
        """Average prediction across all 12 months — comparable to a trailing-12-month live average"""
        from datetime import datetime
        solar_vals = []
        wind_vals = []
        for month in range(1, 13):
            d = datetime(2026, month, 15)
            r = self.predict_with_confidence(lat, lon, d)
            solar_vals.append(r['solar_power_wm2'])
            wind_vals.append(r['wind_power_wm2'])

        avg_solar_raw = sum(solar_vals) / len(solar_vals)
        avg_wind_raw = sum(wind_vals) / len(wind_vals)

        solar_score = min(100, max(0, (avg_solar_raw - 1.46) / 8.11 * 100))
        wind_score = min(100, max(0, (avg_wind_raw - 50) / 500 * 100))

        return {
            'solar': round(solar_score, 1),
            'wind': round(wind_score, 1),
            'hybrid': round((solar_score + wind_score) / 2, 1),
            'solar_power_wm2': round(avg_solar_raw, 2),
            'wind_power_wm2': round(avg_wind_raw, 1)
        }
    
    def _create_prediction_features(self, lat, lon, date):
        """Create feature vector for prediction"""
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        #return [
        #    np.sin(lat_rad), np.cos(lat_rad),
        #    np.sin(lon_rad), np.cos(lon_rad),
        #    np.sin(2 * np.pi * month / 12),
        #    np.cos(2 * np.pi * month / 12),
        #    day_of_year
        #]

        #return pd.DataFrame([[
        #    np.sin(lat_rad), np.cos(lat_rad),
        #    np.sin(lon_rad), np.cos(lon_rad),
        #    np.sin(2 * np.pi * month / 12),
        #    np.cos(2 * np.pi * month / 12),
        #    day_of_year
        #]], columns=['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 'month_sin', 'month_cos', 'day_of_year'])
        
        return np.array([[
            np.sin(lat_rad), np.cos(lat_rad),
            np.sin(lon_rad), np.cos(lon_rad),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            day_of_year
        ]])   

    def find_hotspots(self, bounds, energy_type='hybrid', resolution=0.5):
        """Find clusters of high renewable energy potential"""
        lat_grid = np.arange(bounds['min_lat'], bounds['max_lat'], resolution)
        lon_grid = np.arange(bounds['min_lon'], bounds['max_lon'], resolution)
        
        scores = []
        coordinates = []
        
        print(f"Scanning {len(lat_grid) * len(lon_grid)} points...")
        
        for lat in lat_grid:
            for lon in lon_grid:
                pred = self.predict_location(lat, lon)
                
                if energy_type == 'solar':
                    score = pred['solar']
                elif energy_type == 'wind':
                    score = pred['wind']
                else:
                    score = pred['hybrid']
                
                scores.append(score)
                coordinates.append([lat, lon])
        
        # Find high-scoring clusters
        scores_array = np.array(scores)
        coords_array = np.array(coordinates)
        
        # Filter for high potential (score > 70)
        high_potential = coords_array[scores_array > 70]
        
        if len(high_potential) > 0:
            # Use DBSCAN to find clusters
            clustering = DBSCAN(eps=resolution * 1.5, min_samples=3).fit(high_potential)
            
            hotspots = []
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:
                    cluster_points = high_potential[clustering.labels_ == cluster_id]
                    center = cluster_points.mean(axis=0)
                    
                    # Calculate cluster intensity (average score)
                    cluster_scores = []
                    for point in cluster_points:
                        idx = np.where((coords_array == point).all(axis=1))[0][0]
                        cluster_scores.append(scores_array[idx])
                    
                    hotspots.append({
                        'center_lat': float(center[0]),
                        'center_lon': float(center[1]),
                        'radius_km': len(cluster_points) * resolution * 111,  # Rough estimate
                        'intensity': float(np.mean(cluster_scores)),
                        'area_km2': len(cluster_points) * (resolution * 111) ** 2
                    })
            
            return sorted(hotspots, key=lambda x: x['intensity'], reverse=True)
        
        return []


# Initialize and train (if run directly)
if __name__ == '__main__':
    print("=" * 60)
    print("🤖 TerraVolt ML Model Trainer")
    print("=" * 60)
    
    predictor = RenewableEnergyPredictor()
    
    # Try to load existing models first
    predictor.load_models()
    
    # If models don't exist or you want to retrain, uncomment below
    # print("\n🔄 Training new models...")
    # predictor.train_models()
    
    # Test prediction
    print("\n📊 Test prediction for Upington, Northern Cape (-28.4, 21.2):")
    test_pred = predictor.predict_with_confidence(-28.4, 21.2)
    print(f"   Solar: {test_pred['solar']}/100 (confidence: {test_pred['solar_confidence']}%)")
    print(f"   Wind: {test_pred['wind']}/100 (confidence: {test_pred['wind_confidence']}%)")
    print(f"   Hybrid: {test_pred['hybrid']}/100")
    
    print("\n✅ ML model module ready!") 