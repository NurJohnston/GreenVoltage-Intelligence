import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from database import init_database, SolarHistorical, WindHistorical
import joblib
from datetime import datetime, timedelta

class RenewableEnergyPredictor:
    def __init__(self):
        self.session, self.engine = init_database()
        self.solar_model = None
        self.wind_model = None
        self.scaler = StandardScaler()
    
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
        
        if solar_data:
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
        
        if wind_data:
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
        
        print("Model training complete!")
    
    def predict_location(self, lat, lon, date=None):
        """Predict renewable energy potential for a location"""
        if date is None:
            date = datetime.now()
        
        # Create feature vector
        features = self._create_prediction_features(lat, lon, date)
        
        # Make predictions
        solar_pred = self.solar_model.predict([features])[0] if self.solar_model else 0
        wind_pred = self.wind_model.predict([features])[0] if self.wind_model else 0
        
        # Convert to scores (0-100)
        solar_score = min(100, max(0, (solar_pred - 50) / 250 * 100))
        wind_score = min(100, max(0, (wind_pred - 10) / 500 * 100))  # Wind power density scaling
        
        return {
            'solar': round(solar_score, 1),
            'wind': round(wind_score, 1),
            'hybrid': round((solar_score + wind_score) / 2, 1),
            'solar_power_wm2': round(solar_pred, 1),
            'wind_power_wm2': round(wind_pred, 1)
        }
    
    def _create_prediction_features(self, lat, lon, date):
        """Create feature vector for prediction"""
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        return [
            np.sin(lat_rad), np.cos(lat_rad),
            np.sin(lon_rad), np.cos(lon_rad),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            day_of_year
        ]
    
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

# Initialize and train
if __name__ == '__main__':
    predictor = RenewableEnergyPredictor()
    predictor.train_models()