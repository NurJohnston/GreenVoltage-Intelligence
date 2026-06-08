from database import init_database, SolarHistorical

session, engine = init_database()
records = session.query(SolarHistorical).all()
converted = 0
for r in records:
    if r.irradiance and 0 < r.irradiance < 150:
        r.irradiance = r.irradiance * 11.574
        converted += 1
session.commit()
print(f"Converted {converted} records")