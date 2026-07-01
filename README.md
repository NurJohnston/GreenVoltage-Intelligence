# 🌍 TerraVolt Intelligence

AI-Powered Renewable Energy Optimization Platform for South Africa

Interactive web app that analyzes solar and wind energy potential across South Africa using live data + machine learning.

---

## Features

- Click any location on a map of South Africa
- Live Data: Real-time solar radiation and wind speed
- Energy Analysis: 12-month average from NASA POWER and Open-Meteo
- AI Predictions: ML model trained on 40+ years of SA climate data
- 12-Month Forecast: Seasonal projections for any location
- Heatmaps: Visualize solar/wind/hybrid potential
- Top Locations: Ranked leaderboards
- Ethics Safeguard: Community impact score + acknowledgment checklist

---

## Tech Stack

- Backend: Python, Flask
- ML: scikit-learn, pandas, numpy
- Database: SQLite, SQLAlchemy
- Frontend: HTML, CSS, JavaScript, Leaflet.js, Chart.js

---

## Quick Start

# Clone the repo
git clone https://github.com/NurJohnston/GreenVoltage-Intelligence.git
cd GreenVoltage-Intelligence

# Set up virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
