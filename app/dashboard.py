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

current_data = pd.DataFrame({
    'PM2.5': [185.0],
    'traffic_ratio': [2.4],
    'industry_ratio': [0.8],
    'dust_ratio': [1.6],
    'biomass_ratio': [0.5]
})

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

tab_attribution, tab_map, tab_history, tab_shap = st.tabs([
    "📊 Attribution & Policy", 
    "🗺️ Live Sensor Map",
    "📈 Historical Trends", 
    "🕵️ Model Explainability"
])

# ---------------------------------------------------------------------------
# Tab 1: Attribution & Policy
# ---------------------------------------------------------------------------

with tab_attribution:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Current Causal Attribution")

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
        
        # AI Policy Advisor Text Box
        from src.utils.ai_advisor import generate_policy_recommendation
        
        advisor_text = generate_policy_recommendation(
            weights=mock_weights, 
            current_pm25=current_data['PM2.5'].iloc[0]
        )
        st.info(advisor_text)

    with col2:
        st.subheader("Counterfactual Policy Simulator")
        st.write("Adjust the sliders to simulate the impact of emission reduction policies.")

        traffic_cut = st.slider("Traffic Reduction (%)", 0, 100, 30) / 100.0
        industry_cut = st.slider("Industrial Reduction (%)", 0, 100, 0) / 100.0
        dust_cut = st.slider("Construction Suppression (%)", 0, 100, 20) / 100.0

        interventions = {}
        if traffic_cut > 0: interventions['traffic_ratio'] = -traffic_cut
        if industry_cut > 0: interventions['industry_ratio'] = -industry_cut
        if dust_cut > 0: interventions['dust_ratio'] = -dust_cut

        summary = sim.scenario_summary(current_data, interventions)
        simulated_pm25 = summary['mean_PM2.5_counterfactual']
        pm25_drop = summary['mean_pm25_delta']

        st.metric(
            label="Simulated PM2.5 Level",
            value=f"{simulated_pm25:.1f} µg/m³",
            delta=f"{pm25_drop:.1f} µg/m³",
            delta_color="normal"
        )

        with st.expander("View Raw Simulation Data"):
            result_df = sim.what_if(current_data, interventions)
            st.dataframe(result_df)

# ---------------------------------------------------------------------------
# Tab 2: Live Sensor Map
# ---------------------------------------------------------------------------

with tab_map:
    st.subheader("Live Global PM2.5 Map")
    st.info("Latest PM2.5 observations from OpenAQ. Marker clusters expand as you zoom in.")

    from app.components.map_view import create_sensor_map
    from streamlit_folium import st_folium
    
    @st.cache_data(ttl=600, show_spinner=False)
    def load_live_stations():
        from src.ingestion.aqi_fetcher import OpenAQFetcher
        return OpenAQFetcher().fetch_latest_pm25()

    np.random.seed(42)
    demo_stations = pd.DataFrame({
        'lat': [28.6139, 28.5355, 28.7041, 28.5921], 
        'lon': [77.2090, 77.2641, 77.1025, 77.0460],
        'station': ['Central', 'South', 'North', 'West'],
        'PM2.5': [185.0, 45.0, 110.0, 300.0], 
        'wind_u': np.random.uniform(-10, 10, 4), 
        'wind_v': np.random.uniform(-10, 10, 4)  
    })
    
    try:
        stations = load_live_stations()
        if stations.empty:
            raise RuntimeError("OpenAQ returned no usable PM2.5 readings.")
        st.caption(f"Showing {len(stations):,} recent monitoring observations. Refreshes every 10 minutes.")
    except Exception as error:
        stations = demo_stations
        st.warning(f"Live OpenAQ data is unavailable, so the map is showing demo stations. ({error})")

    sensor_map = create_sensor_map(stations, global_view=True)
    st_folium(sensor_map, width=1200, height=500, returned_objects=[])

    with st.expander("View Station Data"):
        st.dataframe(stations)

# ---------------------------------------------------------------------------
# Tab 3: Historical Trends
# ---------------------------------------------------------------------------
from app.components.charts import create_historical_trend_chart, create_shap_feature_importance_chart

with tab_history:
    st.subheader("30-Day Air Quality Trends")
    st.write("Monitor how PM2.5 levels fluctuate against our safe target limit over time.")
    
    hist_fig = create_historical_trend_chart(pd.DataFrame())
    st.plotly_chart(hist_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Model Explainability (SHAP)
# ---------------------------------------------------------------------------
with tab_shap:
    st.subheader("Meteorological Drivers (SHAP Values)")
    st.write("Understand exactly how weather conditions (Temperature, Wind, Boundary Layer) are impacting pollution levels today. Green bars indicate weather that clears pollution; Red bars indicate weather that traps it.")
    
    shap_fig = create_shap_feature_importance_chart(None, None)
    st.plotly_chart(shap_fig, use_container_width=True)
