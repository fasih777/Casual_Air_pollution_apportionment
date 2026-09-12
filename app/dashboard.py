import numpy as np

# NumPy 2.0+ compatibility shim for legacy scientific packages
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.express as px
import pandas as pd

from src.models.counterfactual import BetaWeights, PolicySimulator

st.set_page_config(page_title="Causal Air Apportionment", layout="wide")
st.title("Micro-Locality Causal Source Apportionment")

# ---------------------------------------------------------------------------
# Mock State Initialization
# ---------------------------------------------------------------------------

# In a production app, this would be loaded from a live API or database
current_data = pd.DataFrame({
    'PM2.5': [185.0],
    'traffic_ratio': [2.4],
    'industry_ratio': [0.8],
    'dust_ratio': [1.6],
    'biomass_ratio': [0.5]
})

# In a production app, these weights would be loaded from the Double ML engine fit
mock_weights = BetaWeights(
    intercept=15.0,
    coefficients={
        'traffic_ratio': 25.0,
        'industry_ratio': 30.0,
        'dust_ratio': 20.0,
        'biomass_ratio': 15.0
    },
    target="PM2.5"
)
sim = PolicySimulator(mock_weights)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

tab_attribution, tab_map = st.tabs(["📊 Attribution & Policy", "🗺️ Live Sensor Map"])

# ---------------------------------------------------------------------------
# Tab 1: Attribution & Policy (existing UI, unchanged in behavior)
# ---------------------------------------------------------------------------

with tab_attribution:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Current Causal Attribution")

        # Calculate the exact mass contributed by each source
        # (weight * proxy_value) for the pie chart
        contributions = {
            'Traffic': mock_weights.coefficients['traffic_ratio'] * current_data['traffic_ratio'].iloc[0],
            'Industry': mock_weights.coefficients['industry_ratio'] * current_data['industry_ratio'].iloc[0],
            'Construction Dust': mock_weights.coefficients['dust_ratio'] * current_data['dust_ratio'].iloc[0],
            'Biomass Burning': mock_weights.coefficients['biomass_ratio'] * current_data['biomass_ratio'].iloc[0],
        }

        fig = px.pie(
            values=list(contributions.values()),
            names=list(contributions.keys()),
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Counterfactual Policy Simulator")
        st.write("Adjust the sliders to simulate the impact of emission reduction policies.")

        traffic_cut = st.slider("Traffic Reduction (%)", 0, 100, 30) / 100.0
        industry_cut = st.slider("Industrial Reduction (%)", 0, 100, 0) / 100.0
        dust_cut = st.slider("Construction Suppression (%)", 0, 100, 20) / 100.0

        # Negative fractional change for cuts (e.g. 30% cut -> -0.30)
        interventions = {}
        if traffic_cut > 0: interventions['traffic_ratio'] = -traffic_cut
        if industry_cut > 0: interventions['industry_ratio'] = -industry_cut
        if dust_cut > 0: interventions['dust_ratio'] = -dust_cut

        # Run the policy simulator
        summary = sim.scenario_summary(current_data, interventions)

        baseline_pm25 = summary['mean_PM2.5_baseline']
        simulated_pm25 = summary['mean_PM2.5_counterfactual']
        pm25_drop = summary['mean_pm25_delta']

        st.metric(
            label="Simulated PM2.5 Level",
            value=f"{simulated_pm25:.1f} µg/m³",
            delta=f"{pm25_drop:.1f} µg/m³",  # Delta is already negative
            delta_color="normal"
        )

        # Optional debugging view
        with st.expander("View Raw Simulation Data"):
            result_df = sim.what_if(current_data, interventions)
            st.dataframe(result_df)

# ---------------------------------------------------------------------------
# Tab 2: Live Sensor Map (placeholder — to be wired up to real sensor feed)
# ---------------------------------------------------------------------------

with tab_map:
    st.subheader("Live Sensor Map")
    st.info(
        "🚧 Placeholder view — this will show real-time readings from the "
        "monitoring station network once the live sensor feed is wired in."
    )

    # Using the new folium map component
    from app.components.map_view import create_sensor_map
    from streamlit_folium import st_folium
    
    # Let's generate a more robust mock dataset with PM2.5 and wind vectors
    # to demonstrate the new capabilities of the map component
    import numpy as np
    np.random.seed(42)
    mock_stations = pd.DataFrame({
        'lat': [28.6139, 28.5355, 28.7041, 28.5921], # Delhi coordinates
        'lon': [77.2090, 77.2641, 77.1025, 77.0460],
        'station': ['Central', 'South', 'North', 'West'],
        'PM2.5': [185.0, 45.0, 110.0, 300.0], # Varying AQI levels
        'wind_u': np.random.uniform(-10, 10, 4), # Random X wind
        'wind_v': np.random.uniform(-10, 10, 4)  # Random Y wind
    })
    
    # Render the map
    sensor_map = create_sensor_map(mock_stations)
    st_folium(sensor_map, width=1200, height=500, returned_objects=[])

    with st.expander("View Station Data"):
        st.dataframe(mock_stations)
