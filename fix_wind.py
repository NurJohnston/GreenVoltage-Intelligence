from database import init_database, WindHistorical

session, engine = init_database()
records = session.query(WindHistorical).all()

for r in records:
    if r.wind_speed:
        r.wind_speed = r.wind_speed / 3.6  # km/h to m/s
        air_density = 1.225
        r.wind_power_density = 0.5 * air_density * (r.wind_speed ** 3)

session.commit()
print("Done")