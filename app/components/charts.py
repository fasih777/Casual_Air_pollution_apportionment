import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_historical_trend_chart(df_history: pd.DataFrame):
    """
    Creates a 30-day historical time-series chart showing PM2.5 levels.
    """
    if df_history.empty:
        # Generate mock 30-day historical data for demonstration
        dates = pd.date_range(end=pd.Timestamp.today(), periods=30)
        np.random.seed(42)
        base_pm25 = np.sin(np.linspace(0, 10, 30)) * 20 + 80
        noise = np.random.normal(0, 10, 30)
        df_history = pd.DataFrame({
            'Date': dates,
            'PM2.5': base_pm25 + noise,
            'Target Limit': [50.0] * 30
        })

    fig = px.line(
        df_history, 
        x='Date', 
        y='PM2.5', 
        title='30-Day PM2.5 Historical Trend',
        markers=True,
        color_discrete_sequence=['#FF6B6B']
    )
    
    # Add target limit line
    fig.add_trace(go.Scatter(
        x=df_history['Date'], 
        y=df_history['Target Limit'],
        mode='lines',
        name='Safe Target Limit',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    fig.update_layout(template="plotly_dark", hovermode="x unified")
    return fig

def create_shap_feature_importance_chart(feature_names, shap_values):
    """
    Creates a bar chart visualizing SHAP feature importances for the confounders (weather).
    """
    if not feature_names or not shap_values:
        # Mock SHAP data for meteorological confounders
        feature_names = ['Temperature', 'Wind Speed (U)', 'Wind Speed (V)', 'PBLH', 'Humidity']
        shap_values = [15.2, -12.4, -8.1, -22.5, 5.3] # Negative means it reduces PM2.5 (e.g. wind clears it)

    df_shap = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value (Impact on PM2.5)': shap_values
    })
    
    # Sort by absolute impact
    df_shap['Abs_Impact'] = df_shap['SHAP Value (Impact on PM2.5)'].abs()
    df_shap = df_shap.sort_values(by='Abs_Impact', ascending=True)
    
    # Color based on positive/negative impact
    df_shap['Color'] = np.where(df_shap['SHAP Value (Impact on PM2.5)'] > 0, '#FF4B4B', '#00CC96')

    fig = px.bar(
        df_shap, 
        x='SHAP Value (Impact on PM2.5)', 
        y='Feature',
        orientation='h',
        title='Meteorological Drivers (SHAP Explainability)',
        color='Color',
        color_discrete_map="identity"
    )
    
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        xaxis_title="Impact on PM2.5 (µg/m³)",
        yaxis_title="Weather Variable"
    )
    
    # Add vertical zero line
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="white")
    
    return fig
