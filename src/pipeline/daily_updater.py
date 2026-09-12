"""
daily_updater.py

End-to-end pipeline script to fetch live data, compute features, update the local dataset,
and retrain the Causal Model.

Run this daily via Windows Task Scheduler or a Cron job.
"""

import os
import logging
from datetime import datetime, timedelta
import pandas as pd

from src.ingestion.aqi_fetcher import OpenAQFetcher
from src.ingestion.weather_fetcher import OpenMeteoFetcher
from src.features.chemical_ratios import compute_all_ratios
from src.features.wind_vectoring import compute_wind_vectors
from src.models.double_ml_engine import CausalSourceApportionmentEngine
from src.models.counterfactual import BetaWeights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Hardcoded for demonstration (New Delhi coordinates)
LAT, LON = 28.6139, 77.2090
OPENAQ_LOCATION_ID = 8118 # Example location ID
PROCESSED_DATA_PATH = "data/processed/live_data.csv"
WEIGHTS_PATH = "data/processed/latest_weights.json"

def run_daily_update():
    logger.info("Starting Daily Data Update Pipeline...")
    
    # 1. Determine date range (Last 24 hours)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=1)
    date_str_end = end_date.strftime("%Y-%m-%d")
    date_str_start = start_date.strftime("%Y-%m-%d")
    
    logger.info(f"Fetching data from {date_str_start} to {date_str_end}")
    
    # 2. Fetch Data
    aqi_fetcher = OpenAQFetcher()
    weather_fetcher = OpenMeteoFetcher()
    
    logger.info("Fetching Air Quality data...")
    try:
        df_aqi = aqi_fetcher.fetch_data(
            location_id=OPENAQ_LOCATION_ID,
            date_from=start_date.isoformat() + "Z",
            date_to=end_date.isoformat() + "Z"
        )
    except Exception as e:
        logger.error(f"Failed to fetch AQI data: {e}")
        df_aqi = pd.DataFrame()

    logger.info("Fetching Weather data...")
    try:
        df_weather = weather_fetcher.fetch_data(
            lat=LAT, lon=LON, 
            start_date=date_str_start, 
            end_date=date_str_end
        )
    except Exception as e:
        logger.error(f"Failed to fetch Weather data: {e}")
        df_weather = pd.DataFrame()
        
    if df_aqi.empty or df_weather.empty:
        logger.warning("Missing data from APIs. Aborting update for today.")
        return

    # 3. Merge Data (Assuming 'date_utc' and 'time' align roughly by hour)
    # This is a simple merge for demonstration. In production, ensure tz alignment.
    df_aqi['date_utc'] = pd.to_datetime(df_aqi['date_utc']).dt.tz_localize(None)
    df_weather['time'] = pd.to_datetime(df_weather['time']).dt.tz_localize(None)
    
    df_merged = pd.merge(df_aqi, df_weather, left_on='date_utc', right_on='time', how='inner')
    
    # Ensure column names map to what features expect
    # The OpenAQ fetcher creates columns for 'pm25', 'pm10', 'no2', 'co', 'so2'.
    # We rename them to uppercase to match the chemical_ratios.py defaults.
    df_merged = df_merged.rename(columns={
        'pm25': 'PM2.5', 'pm10': 'PM10', 'no2': 'NO2', 'co': 'CO', 'so2': 'SO2'
    })

    # Mock Potassium (K) data since it's rarely available live via OpenAQ
    df_merged['K'] = 0.5 
    
    # 4. Feature Engineering
    logger.info("Computing Chemical Ratios & Wind Vectors...")
    df_merged = compute_all_ratios(df_merged)
    df_merged = compute_wind_vectors(df_merged)
    
    # 5. Save/Append Data
    if os.path.exists(PROCESSED_DATA_PATH):
        df_existing = pd.read_csv(PROCESSED_DATA_PATH)
        df_combined = pd.concat([df_existing, df_merged]).drop_duplicates(subset=['time'])
    else:
        df_combined = df_merged
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_combined.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Dataset updated. Total rows: {len(df_combined)}")
    
    # 6. Retrain Causal Model if we have enough data (e.g., > 100 rows)
    if len(df_combined) >= 100:
        logger.info("Sufficient data available. Retraining Causal Double ML Engine...")
        engine = CausalSourceApportionmentEngine()
        try:
            weights = engine.fit(df_combined)
            weights.to_json(WEIGHTS_PATH)
            logger.info(f"Model retrained. New weights saved to {WEIGHTS_PATH}")
        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")

if __name__ == "__main__":
    run_daily_update()
