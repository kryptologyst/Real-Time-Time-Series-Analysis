"""Streamlit web interface for time series analysis."""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, Any, Optional
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_utils import SyntheticDataGenerator, TimeSeriesConfig, DataPreprocessor, RollingWindowAnalyzer
from models import ARIMAForecaster, ProphetForecaster, AnomalyDetector, ModelEnsemble, ModelConfig
from visualization import TimeSeriesVisualizer, InteractiveVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Real-Time Time Series Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Real-Time Time Series Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    data_length = st.sidebar.slider("Data Length", 100, 2000, 1000)
    trend_strength = st.sidebar.slider("Trend Strength", 0.0, 1.0, 0.1)
    seasonality_strength = st.sidebar.slider("Seasonality Strength", 0.0, 1.0, 0.5)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
    anomaly_probability = st.sidebar.slider("Anomaly Probability", 0.0, 0.2, 0.05)
    
    # Model parameters
    st.sidebar.subheader("Model Configuration")
    forecast_steps = st.sidebar.slider("Forecast Steps", 10, 100, 30)
    window_size = st.sidebar.slider("Rolling Window Size", 10, 100, 30)
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    enable_arima = st.sidebar.checkbox("Enable ARIMA", value=True)
    enable_prophet = st.sidebar.checkbox("Enable Prophet", value=True)
    enable_anomaly_detection = st.sidebar.checkbox("Enable Anomaly Detection", value=True)
    enable_real_time = st.sidebar.checkbox("Enable Real-Time Visualization", value=False)
    
    # Initialize session state
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'models_fitted' not in st.session_state:
        st.session_state.models_fitted = False
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔮 Forecasting", 
        "🚨 Anomaly Detection", 
        "⚡ Real-Time Analysis", 
        "📈 Interactive Plots"
    ])
    
    with tab1:
        data_overview_tab(data_length, trend_strength, seasonality_strength, 
                         noise_level, anomaly_probability)
    
    with tab2:
        forecasting_tab(forecast_steps, enable_arima, enable_prophet)
    
    with tab3:
        anomaly_detection_tab(enable_anomaly_detection)
    
    with tab4:
        real_time_tab(window_size, enable_real_time)
    
    with tab5:
        interactive_plots_tab()


def data_overview_tab(data_length: int, trend_strength: float, seasonality_strength: float,
                     noise_level: float, anomaly_probability: float):
    """Data overview tab."""
    
    st.header("📊 Data Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Generate New Data", key="generate_data"):
            # Generate synthetic data
            config = TimeSeriesConfig(
                length=data_length,
                trend_strength=trend_strength,
                seasonality_strength=seasonality_strength,
                noise_level=noise_level,
                anomaly_probability=anomaly_probability
            )
            
            generator = SyntheticDataGenerator(config)
            series = generator.generate_series(seed=42)
            
            # Store in session state
            st.session_state.series = series
            st.session_state.data_generated = True
            
            st.success("Data generated successfully!")
    
    with col2:
        if st.session_state.data_generated:
            st.metric("Data Points", len(st.session_state.series))
            st.metric("Mean", f"{st.session_state.series.mean():.3f}")
            st.metric("Std Dev", f"{st.session_state.series.std():.3f}")
    
    if st.session_state.data_generated:
        series = st.session_state.series
        
        # Plot the time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name='Time Series',
            line=dict(width=2, color='blue')
        ))
        
        fig.update_layout(
            title="Generated Time Series",
            xaxis_title="Time",
            yaxis_title="Value",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data statistics
        st.subheader("Data Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min", f"{series.min():.3f}")
        with col2:
            st.metric("Max", f"{series.max():.3f}")
        with col3:
            st.metric("Skewness", f"{series.skew():.3f}")
        with col4:
            st.metric("Kurtosis", f"{series.kurtosis():.3f}")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(series.head(10))


def forecasting_tab(forecast_steps: int, enable_arima: bool, enable_prophet: bool):
    """Forecasting tab."""
    
    st.header("🔮 Forecasting")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the Data Overview tab.")
        return
    
    series = st.session_state.series
    
    # Split data for training and testing
    split_point = int(len(series) * 0.8)
    train_series = series[:split_point]
    test_series = series[split_point:]
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Run Forecasting Models", key="run_forecasting"):
            models = []
            forecasts = {}
            
            # ARIMA model
            if enable_arima:
                try:
                    arima_config = ModelConfig()
                    arima_model = ARIMAForecaster(arima_config)
                    arima_model.fit(train_series)
                    arima_forecast = arima_model.predict(forecast_steps)
                    forecasts['ARIMA'] = arima_forecast
                    models.append(arima_model)
                    st.success("ARIMA model fitted successfully!")
                except Exception as e:
                    st.error(f"ARIMA model failed: {e}")
            
            # Prophet model
            if enable_prophet:
                try:
                    prophet_config = ModelConfig()
                    prophet_model = ProphetForecaster(prophet_config)
                    prophet_model.fit(train_series)
                    prophet_forecast = prophet_model.predict(forecast_steps)
                    forecasts['Prophet'] = prophet_forecast
                    models.append(prophet_model)
                    st.success("Prophet model fitted successfully!")
                except Exception as e:
                    st.error(f"Prophet model failed: {e}")
            
            # Store results
            st.session_state.forecasts = forecasts
            st.session_state.models = models
            st.session_state.models_fitted = True
            
    with col2:
        if st.session_state.models_fitted:
            st.metric("Models Fitted", len(st.session_state.models))
            st.metric("Forecast Steps", forecast_steps)
    
    if st.session_state.models_fitted:
        forecasts = st.session_state.forecasts
        
        # Create forecast plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=train_series.index,
            y=train_series.values,
            mode='lines',
            name='Training Data',
            line=dict(color='blue', width=2)
        ))
        
        # Test data
        fig.add_trace(go.Scatter(
            x=test_series.index,
            y=test_series.values,
            mode='lines',
            name='Test Data',
            line=dict(color='green', width=2)
        ))
        
        # Forecasts
        colors = ['red', 'orange', 'purple', 'brown']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title="Time Series Forecasting",
            xaxis_title="Time",
            yaxis_title="Value",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model comparison
        if len(forecasts) > 1:
            st.subheader("Model Comparison")
            
            # Calculate metrics
            comparison_data = []
            for model_name, forecast in forecasts.items():
                # Use test data for evaluation
                test_forecast = forecast[:len(test_series)]
                mae = np.mean(np.abs(test_series.values - test_forecast.values))
                mse = np.mean((test_series.values - test_forecast.values) ** 2)
                rmse = np.sqrt(mse)
                
                comparison_data.append({
                    'Model': model_name,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)


def anomaly_detection_tab(enable_anomaly_detection: bool):
    """Anomaly detection tab."""
    
    st.header("🚨 Anomaly Detection")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the Data Overview tab.")
        return
    
    series = st.session_state.series
    
    col1, col2 = st.columns(2)
    
    with col1:
        contamination = st.slider("Contamination", 0.01, 0.2, 0.1)
        
        if st.button("Detect Anomalies", key="detect_anomalies"):
            if enable_anomaly_detection:
                try:
                    config = ModelConfig(contamination=contamination)
                    detector = AnomalyDetector(config)
                    detector.fit(series)
                    anomalies = detector.detect_anomalies(series)
                    
                    st.session_state.anomalies = anomalies
                    st.success("Anomaly detection completed!")
                except Exception as e:
                    st.error(f"Anomaly detection failed: {e}")
            else:
                st.warning("Anomaly detection is disabled.")
    
    with col2:
        if 'anomalies' in st.session_state:
            num_anomalies = st.session_state.anomalies.sum()
            anomaly_rate = num_anomalies / len(series) * 100
            
            st.metric("Anomalies Detected", num_anomalies)
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    
    if 'anomalies' in st.session_state:
        anomalies = st.session_state.anomalies
        
        # Create anomaly plot
        fig = go.Figure()
        
        # Normal data
        normal_data = series[~anomalies]
        fig.add_trace(go.Scatter(
            x=normal_data.index,
            y=normal_data.values,
            mode='lines',
            name='Normal',
            line=dict(color='blue', width=2)
        ))
        
        # Anomaly data
        anomaly_data = series[anomalies]
        fig.add_trace(go.Scatter(
            x=anomaly_data.index,
            y=anomaly_data.values,
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=8)
        ))
        
        fig.update_layout(
            title="Anomaly Detection Results",
            xaxis_title="Time",
            yaxis_title="Value",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)


def real_time_tab(window_size: int, enable_real_time: bool):
    """Real-time analysis tab."""
    
    st.header("⚡ Real-Time Analysis")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the Data Overview tab.")
        return
    
    series = st.session_state.series
    
    if enable_real_time:
        st.subheader("Rolling Window Statistics")
        
        # Initialize rolling analyzer
        analyzer = RollingWindowAnalyzer(window_size)
        
        # Process data in chunks to simulate real-time
        chunk_size = 50
        stats_history = []
        
        for i in range(0, len(series), chunk_size):
            chunk = series.iloc[i:i+chunk_size]
            for value in chunk:
                stats = analyzer.add_data_point(value)
                stats_history.append(stats.copy())
        
        # Convert to DataFrame for visualization
        stats_df = pd.DataFrame(stats_history)
        
        # Plot rolling statistics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rolling Mean', 'Rolling Std', 'Rolling Min/Max', 'Current Value'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Rolling mean
        fig.add_trace(
            go.Scatter(x=stats_df.index, y=stats_df['mean'], name='Mean'),
            row=1, col=1
        )
        
        # Rolling std
        fig.add_trace(
            go.Scatter(x=stats_df.index, y=stats_df['std'], name='Std'),
            row=1, col=2
        )
        
        # Rolling min/max
        fig.add_trace(
            go.Scatter(x=stats_df.index, y=stats_df['min'], name='Min'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=stats_df.index, y=stats_df['max'], name='Max'),
            row=2, col=1
        )
        
        # Current value
        fig.add_trace(
            go.Scatter(x=stats_df.index, y=stats_df['current'], name='Current'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Current statistics
        current_stats = analyzer.get_current_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Mean", f"{current_stats['mean']:.3f}")
        with col2:
            st.metric("Current Std", f"{current_stats['std']:.3f}")
        with col3:
            st.metric("Current Min", f"{current_stats['min']:.3f}")
        with col4:
            st.metric("Current Max", f"{current_stats['max']:.3f}")
    
    else:
        st.info("Enable real-time visualization in the sidebar to see rolling statistics.")


def interactive_plots_tab():
    """Interactive plots tab."""
    
    st.header("📈 Interactive Plots")
    
    if not st.session_state.data_generated:
        st.warning("Please generate data first in the Data Overview tab.")
        return
    
    series = st.session_state.series
    
    # Plot options
    plot_type = st.selectbox(
        "Select Plot Type",
        ["Time Series", "Decomposition", "Distribution", "Autocorrelation"]
    )
    
    if plot_type == "Time Series":
        # Basic time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name='Time Series',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title="Interactive Time Series Plot",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Decomposition":
        # Seasonal decomposition
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            decomposition = seasonal_decompose(series.dropna(), model='additive')
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
                vertical_spacing=0.05
            )
            
            # Original
            fig.add_trace(
                go.Scatter(x=series.index, y=series.values, name='Original'),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name='Trend'),
                row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name='Seasonal'),
                row=3, col=1
            )
            
            # Residual
            fig.add_trace(
                go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name='Residual'),
                row=4, col=1
            )
            
            fig.update_layout(height=800, showlegend=False, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.error("statsmodels is required for decomposition plots")
    
    elif plot_type == "Distribution":
        # Distribution plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=series.values,
            nbinsx=50,
            name='Distribution'
        ))
        
        fig.update_layout(
            title="Value Distribution",
            xaxis_title="Value",
            yaxis_title="Frequency",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Autocorrelation":
        # Autocorrelation plot
        try:
            from statsmodels.tsa.stattools import acf
            
            autocorr = acf(series.dropna(), nlags=50)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(autocorr))),
                y=autocorr,
                mode='lines+markers',
                name='Autocorrelation'
            ))
            
            fig.update_layout(
                title="Autocorrelation Function",
                xaxis_title="Lag",
                yaxis_title="Autocorrelation",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except ImportError:
            st.error("statsmodels is required for autocorrelation plots")


if __name__ == "__main__":
    main()
