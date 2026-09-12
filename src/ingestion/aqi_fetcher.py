import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

class OpenAQFetcher:
    """Fetches air quality data from OpenAQ API."""
    
    BASE_URL = "https://api.openaq.org/v2/measurements"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.headers = {"X-API-Key": self.api_key} if self.api_key else {}

    def fetch_data(self, 
                   location_id: int, 
                   date_from: str, 
                   date_to: str, 
                   parameters: List[str] = ["pm25", "pm10", "no2", "co", "so2"]) -> pd.DataFrame:
        """
        Fetch hourly records for specified pollutants.
        
        Args:
            location_id: OpenAQ location ID.
            date_from: Start date in ISO format (e.g., '2023-01-01T00:00:00Z').
            date_to: End date in ISO format (e.g., '2023-01-02T00:00:00Z').
            parameters: List of pollutants to fetch.
        
        Returns:
            DataFrame containing the measurements.
        """
        all_results = []
        
        for param in parameters:
            params = {
                "location_id": location_id,
                "parameter": param,
                "date_from": date_from,
                "date_to": date_to,
                "limit": 10000,
                "temporal": "hourly"
            }
            
            response = requests.get(self.BASE_URL, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            if "results" in data:
                all_results.extend(data["results"])
                
        if not all_results:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_results)
        
        # Clean up dataframe
        if not df.empty:
            df['date_utc'] = pd.to_datetime(df['date'].apply(lambda x: x['utc']))
            df['date_local'] = pd.to_datetime(df['date'].apply(lambda x: x['local']))
            df = df[['locationId', 'location', 'parameter', 'value', 'unit', 'date_utc', 'date_local']]
            
            # Pivot to wide format (parameters as columns)
            df = df.pivot_table(
                index=['locationId', 'location', 'date_utc', 'date_local'],
                columns='parameter',
                values='value'
            ).reset_index()
            
        return df

if __name__ == "__main__":
    # Example usage
    fetcher = OpenAQFetcher()
    # Using a dummy location_id for testing; replace with actual OpenAQ location ID
    # print(fetcher.fetch_data(location_id=8118, date_from="2024-01-01", date_to="2024-01-02"))
