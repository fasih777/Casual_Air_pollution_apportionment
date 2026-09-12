import requests
import pandas as pd
from typing import Dict, Any

class OpenMeteoFetcher:
    """Fetches meteorological data from Open-Meteo API."""
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def fetch_data(self, 
                   lat: float, 
                   lon: float, 
                   start_date: str, 
                   end_date: str) -> pd.DataFrame:
        """
        Pull hourly temperature, relative humidity, wind speed, wind direction, and PBLH.
        
        Args:
            lat: Latitude of the location.
            lon: Longitude of the location.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            
        Returns:
            DataFrame with hourly meteorological data.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "boundary_layer_height"],
            "timezone": "auto"
        }
        
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if "hourly" not in data:
            return pd.DataFrame()
            
        hourly_data = data["hourly"]
        df = pd.DataFrame({
            "time": pd.to_datetime(hourly_data["time"]),
            "temperature_2m": hourly_data["temperature_2m"],
            "relative_humidity_2m": hourly_data["relative_humidity_2m"],
            "wind_speed_10m": hourly_data["wind_speed_10m"],
            "wind_direction_10m": hourly_data["wind_direction_10m"],
            "pblh": hourly_data["boundary_layer_height"]
        })
        
        # Rename columns to standard names
        df = df.rename(columns={
            "temperature_2m": "temp",
            "relative_humidity_2m": "humidity",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction"
        })
        
        return df

if __name__ == "__main__":
    # Example usage
    fetcher = OpenMeteoFetcher()
    # New Delhi coordinates
    # df = fetcher.fetch_data(lat=28.6139, lon=77.2090, start_date="2024-01-01", end_date="2024-01-02")
    # print(df.head())
