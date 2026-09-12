import requests
import pandas as pd
from typing import Optional

class SatelliteProxyFetcher:
    """Fetches Fire Radiative Power (FRP) from NASA FIRMS API."""
    
    BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def fetch_data(self, 
                   lat: float, 
                   lon: float, 
                   radius_km: int, 
                   date: str, 
                   source: str = "VIIRS_SNPP_NRT",
                   day_range: int = 1) -> pd.DataFrame:
        """
        Extract regional FRP within a radius (approximated as a bounding box here for FIRMS API).
        
        Args:
            lat: Center latitude.
            lon: Center longitude.
            radius_km: Radius in km (approximated).
            date: Date in YYYY-MM-DD format.
            source: satellite source (e.g., VIIRS_SNPP_NRT, MODIS_NRT).
            day_range: Number of days (1-10).
            
        Returns:
            DataFrame with hotspot data.
        """
        # Note: NASA FIRMS API takes a bounding box (west, south, east, north)
        # We perform a rough approximation of the bounding box based on radius.
        # 1 degree is approx 111 km.
        offset = radius_km / 111.0
        
        bbox = f"{lon - offset},{lat - offset},{lon + offset},{lat + offset}"
        
        url = f"{self.BASE_URL}/{self.api_key}/{source}/{bbox}/{day_range}/{date}"
        
        try:
            # FIRMS returns CSV data
            df = pd.read_csv(url)
            return df
        except Exception as e:
            print(f"Error fetching NASA FIRMS data: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    # fetcher = SatelliteProxyFetcher(api_key="YOUR_NASA_FIRMS_MAP_KEY")
    # df = fetcher.fetch_data(lat=28.6139, lon=77.2090, radius_km=50, date="2024-01-01")
    pass
