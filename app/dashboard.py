import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

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
        delta=f"{pm25_drop:.1f} µg/m³", # Delta is already negative
        delta_color="normal"
    )
    
    # Optional debugging view
    with st.expander("View Raw Simulation Data"):
        result_df = sim.what_if(current_data, interventions)
        st.dataframe(result_df)
