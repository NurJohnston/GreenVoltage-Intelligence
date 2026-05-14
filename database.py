from sqlalchemy import create_engine, Column, Float, String, Date, DateTime, Integer, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class SolarHistorical(Base):
    __tablename__ = 'solar_historical'
    id = Column(Integer, primary_key=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    date = Column(Date, nullable=False)
    irradiance = Column(Float)  # W/m²
    uv_index = Column(Float)
    cloud_cover = Column(Float)
    
    __table_args__ = (
        Index('idx_solar_location_date', 'lat', 'lon', 'date'),
    )

class WindHistorical(Base):
    __tablename__ = 'wind_historical'
    id = Column(Integer, primary_key=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    date = Column(Date, nullable=False)
    wind_speed = Column(Float)  # m/s
    wind_direction = Column(Float)  # degrees
    wind_gust = Column(Float)  # m/s
    wind_power_density = Column(Float)  # W/m² (calculated)
    
    __table_args__ = (
        Index('idx_wind_location_date', 'lat', 'lon', 'date'),
    )

class PredictionCache(Base):
    __tablename__ = 'prediction_cache'
    id = Column(Integer, primary_key=True)
    lat = Column(Float)
    lon = Column(Float)
    energy_type = Column(String)
    predicted_score = Column(Float)
    confidence = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database setup
def init_database():
    database_url = os.getenv('DATABASE_URL', 'sqlite:///renewable_data.db')
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    # Return a session instance, not the sessionmaker class
    return Session(), engine

# Helper function to store data
def store_weather_data(session, lat, lon, date, solar_data=None, wind_data=None):
    """Store weather data in database"""
    try:
        if solar_data and solar_data.get('irradiance') is not None:
            solar_record = SolarHistorical(
                lat=lat, 
                lon=lon, 
                date=date,
                irradiance=solar_data.get('irradiance'),
                uv_index=solar_data.get('uv_index'),
                cloud_cover=solar_data.get('cloud_cover')
            )
            session.add(solar_record)
        
        if wind_data and wind_data.get('wind_speed') is not None:
            # Calculate wind power density: P = 0.5 * air_density * v³
            air_density = 1.225  # kg/m³ at sea level
            wind_power = 0.5 * air_density * (wind_data.get('wind_speed', 0) ** 3)
            
            wind_record = WindHistorical(
                lat=lat, 
                lon=lon, 
                date=date,
                wind_speed=wind_data.get('wind_speed'),
                wind_direction=wind_data.get('wind_direction', 0),
                wind_gust=wind_data.get('wind_gust', 0),
                wind_power_density=wind_power
            )
            session.add(wind_record)
        
        session.commit()
    except Exception as e:
        print(f"Error storing data: {e}")
        session.rollback()