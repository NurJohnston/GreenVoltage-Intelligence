"""
Train ML models on REAL historical data from database
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from database import init_database, SolarHistorical, WindHistorical
from datetime import datetime

print("=" * 60)
print("🤖 Training ML Models on REAL South African Data")
print("=" * 60)

# Initialize database
session, engine = init_database()

# Query solar data
print("\n📊 Loading solar data from database...")
solar_query = session.query(SolarHistorical)
solar_df = pd.read_sql(solar_query.statement, engine)

# Query wind data
print("📊 Loading wind data from database...")
wind_query = session.query(WindHistorical)
wind_df = pd.read_sql(wind_query.statement, engine)

print(f"\n✅ Loaded {len(solar_df)} solar records")
print(f"✅ Loaded {len(wind_df)} wind records")

if len(solar_df) == 0 or len(wind_df) == 0:
    print("\n❌ No data found! Run data_collector.py first.")
    exit()

# Feature engineering
print("\n🔧 Engineering features...")

def engineer_features(df, target_col):
    """Create features for ML model"""
    if df.empty:
        return None, None
    
    # Convert date to features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
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
    
    # Features list
    features = ['lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 
                'month_sin', 'month_cos', 'day_of_year']
    
    X = df[features]
    y = df[target_col]
    
    return X, y

# Solar model
print("\n☀️ Training Solar Model...")
X_solar, y_solar = engineer_features(solar_df, 'irradiance')

if X_solar is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_solar, y_solar, test_size=0.2, random_state=42
    )
    
    solar_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    solar_model.fit(X_train, y_train)
    
    # Evaluate
    train_score = solar_model.score(X_train, y_train)
    test_score = solar_model.score(X_test, y_test)
    
    print(f"  Training R²: {train_score:.3f}")
    print(f"  Test R²: {test_score:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(solar_model, X_solar, y_solar, cv=5)
    print(f"  Cross-validation R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Save model
    joblib.dump(solar_model, 'solar_model.pkl')
    print("  ✅ Saved to solar_model.pkl")
else:
    print("  ❌ No solar data available")

# Wind model
print("\n💨 Training Wind Model...")
X_wind, y_wind = engineer_features(wind_df, 'wind_power_density')

if X_wind is not None:
    X_train, X_test, y_train, y_test = train_test_split(
        X_wind, y_wind, test_size=0.2, random_state=42
    )
    
    wind_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    wind_model.fit(X_train, y_train)
    
    # Evaluate
    train_score = wind_model.score(X_train, y_train)
    test_score = wind_model.score(X_test, y_test)
    
    print(f"  Training R²: {train_score:.3f}")
    print(f"  Test R²: {test_score:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(wind_model, X_wind, y_wind, cv=5)
    print(f"  Cross-validation R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Save model
    joblib.dump(wind_model, 'wind_model.pkl')
    print("  ✅ Saved to wind_model.pkl")
else:
    print("  ❌ No wind data available")

print("\n" + "=" * 60)
print("🎉 ML Training Complete! Models are ready for hybrid deployment.")
print("=" * 60)

# Test prediction
print("\n📊 Test prediction for Upington (-28.4, 21.2):")

def predict_location(lat, lon, month=1, day=15):
    """Test prediction function"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    features = np.array([[
        np.sin(lat_rad), np.cos(lat_rad),
        np.sin(lon_rad), np.cos(lon_rad),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        day
    ]])
    
    solar_model = joblib.load('solar_model.pkl')
    wind_model = joblib.load('wind_model.pkl')
    
    solar_pred = solar_model.predict(features)[0]
    wind_pred = wind_model.predict(features)[0]
    
    solar_score = min(100, max(0, (solar_pred - 50) / 250 * 100))
    wind_score = min(100, max(0, (wind_pred - 10) / 500 * 100))
    
    print(f"  Solar: {solar_score:.1f}/100 ({solar_pred:.1f} W/m²)")
    print(f"  Wind: {wind_score:.1f}/100 ({wind_pred:.1f} W/m²)")
    print(f"  Hybrid: {(solar_score + wind_score) / 2:.1f}/100")