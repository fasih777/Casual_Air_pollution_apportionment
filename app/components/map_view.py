import folium
import pandas as pd
import numpy as np

def get_color_for_pm25(val):
    """Return standard AQI color mapping for PM2.5"""
    if val <= 12.0: return 'green'      # Good
    elif val <= 35.4: return 'lightgreen' # Moderate
    elif val <= 55.4: return 'orange'     # USG
    elif val <= 150.4: return 'red'       # Unhealthy
    elif val <= 250.4: return 'purple'    # Very Unhealthy
    else: return 'darkred'                # Hazardous

def create_sensor_map(df_stations: pd.DataFrame) -> folium.Map:
    """
    Creates a Folium map visualizing sensor locations, their PM2.5 levels,
    and optional wind vectors.
    
    Args:
        df_stations: DataFrame containing 'lat', 'lon', 'station', 'PM2.5', 
                     and optionally 'wind_u', 'wind_v'
                     
    Returns:
        A Folium Map object.
    """
    if df_stations.empty:
        # Default to New Delhi coordinates if no data
        m = folium.Map(location=[28.6139, 77.2090], zoom_start=11, tiles="CartoDB dark_matter")
        return m
        
    # Center map on the mean coordinates
    center_lat = df_stations['lat'].mean()
    center_lon = df_stations['lon'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB dark_matter")
    
    for _, row in df_stations.iterrows():
        lat, lon = row['lat'], row['lon']
        name = row['station']
        pm25 = row.get('PM2.5', 0)
        
        # Determine color based on PM2.5 level
        color = get_color_for_pm25(pm25)
        
        # HTML Popup with data
        popup_html = f"<b>{name}</b><br>PM2.5: {pm25:.1f} µg/m³"
        
        if 'wind_u' in row and 'wind_v' in row:
            u, v = row['wind_u'], row['wind_v']
            wind_speed = np.sqrt(u**2 + v**2)
            popup_html += f"<br>Wind: {wind_speed:.1f} m/s"
            
            # Simple line to indicate wind direction (pointing away from sensor)
            # Roughly convert m/s to a small lat/lon offset for visualization
            wind_lat_end = lat + (v * 0.001)
            wind_lon_end = lon + (u * 0.001)
            
            folium.PolyLine(
                locations=[(lat, lon), (wind_lat_end, wind_lon_end)],
                color="white",
                weight=2,
                opacity=0.7,
                dash_array='5, 5'
            ).add_to(m)
            
        folium.CircleMarker(
            location=(lat, lon),
            radius=min(max(pm25 / 10, 5), 25), # Scale radius dynamically, clamped
            popup=folium.Popup(popup_html, max_width=200),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)
        
    return m
