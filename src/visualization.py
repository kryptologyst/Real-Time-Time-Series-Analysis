"""Visualization utilities for time series analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, List, Dict, Any, Tuple
import logging
from collections import deque

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("plotly not available. Interactive plots will be disabled.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logging.warning("seaborn not available. Some plot styles will be disabled.")

logger = logging.getLogger(__name__)


class TimeSeriesVisualizer:
    """Comprehensive time series visualization utilities."""
    
    def __init__(self, style: str = "default"):
        """Initialize visualizer with specified style.
        
        Args:
            style: Plot style ("default", "seaborn", "plotly")
        """
        self.style = style
        self._setup_style()
        
    def _setup_style(self) -> None:
        """Setup plotting style."""
        if self.style == "seaborn" and SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            plt.style.use("seaborn-v0_8")
        elif self.style == "plotly" and PLOTLY_AVAILABLE:
            pass  # Plotly handles styling differently
        else:
            plt.style.use("default")
    
    def plot_time_series(
        self,
        series: pd.Series,
        title: str = "Time Series",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot a single time series.
        
        Args:
            series: Time series data
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(series.index, series.values, linewidth=1.5, alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_decomposition(
        self,
        series: pd.Series,
        model: str = "additive",
        period: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot seasonal decomposition.
        
        Args:
            series: Time series data
            model: Decomposition model ("additive" or "multiplicative")
            period: Seasonal period
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            decomposition = seasonal_decompose(
                series.dropna(),
                model=model,
                period=period
            )
            
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            
            # Original series
            axes[0].plot(series.index, series.values, linewidth=1.5)
            axes[0].set_title("Original", fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(decomposition.trend.index, decomposition.trend.values, linewidth=1.5)
            axes[1].set_title("Trend", fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, linewidth=1.5)
            axes[2].set_title("Seasonal", fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            axes[3].plot(decomposition.resid.index, decomposition.resid.values, linewidth=1.5)
            axes[3].set_title("Residual", fontweight='bold')
            axes[3].grid(True, alpha=0.3)
            
            plt.suptitle(f"Seasonal Decomposition ({model})", fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            return fig
            
        except ImportError:
            logger.error("statsmodels required for decomposition plots")
            raise
    
    def plot_forecast(
        self,
        historical: pd.Series,
        forecast: pd.Series,
        confidence_intervals: Optional[Dict[str, pd.Series]] = None,
        title: str = "Time Series Forecast",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot historical data with forecast.
        
        Args:
            historical: Historical time series data
            forecast: Forecasted values
            confidence_intervals: Optional confidence intervals
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical data
        ax.plot(historical.index, historical.values, 
                label="Historical", linewidth=1.5, alpha=0.8)
        
        # Plot forecast
        ax.plot(forecast.index, forecast.values, 
                label="Forecast", linewidth=2, alpha=0.9)
        
        # Plot confidence intervals if provided
        if confidence_intervals:
            for name, ci in confidence_intervals.items():
                ax.fill_between(ci.index, ci.values, alpha=0.3, label=name)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_anomalies(
        self,
        series: pd.Series,
        anomalies: pd.Series,
        title: str = "Anomaly Detection",
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot time series with detected anomalies.
        
        Args:
            series: Time series data
            anomalies: Boolean series indicating anomalies
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot normal data
        normal_data = series[~anomalies]
        ax.plot(normal_data.index, normal_data.values, 
                'b-', label="Normal", linewidth=1.5, alpha=0.8)
        
        # Plot anomalies
        anomaly_data = series[anomalies]
        ax.scatter(anomaly_data.index, anomaly_data.values, 
                   color='red', s=50, label="Anomaly", alpha=0.8)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multiple_series(
        self,
        series_dict: Dict[str, pd.Series],
        title: str = "Multiple Time Series",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot multiple time series on the same plot.
        
        Args:
            series_dict: Dictionary of series name -> series data
            title: Plot title
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(series_dict)))
        
        for i, (name, series) in enumerate(series_dict.items()):
            ax.plot(series.index, series.values, 
                    label=name, linewidth=1.5, alpha=0.8, color=colors[i])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class RealTimeVisualizer:
    """Real-time visualization with matplotlib animation."""
    
    def __init__(self, window_size: int = 50, update_interval: int = 100):
        """Initialize real-time visualizer.
        
        Args:
            window_size: Size of the rolling window
            update_interval: Animation update interval in milliseconds
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self.data_buffer = deque(maxlen=window_size)
        self.stats_buffer = deque(maxlen=window_size)
        
    def create_animation(
        self,
        data_stream: np.ndarray,
        title: str = "Real-Time Time Series Analysis"
    ) -> animation.FuncAnimation:
        """Create real-time animation.
        
        Args:
            data_stream: Stream of data points
            title: Animation title
            
        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Initialize empty lines
        line_data, = ax.plot([], [], label="Signal", linewidth=2, alpha=0.8)
        line_ma, = ax.plot([], [], label="Rolling Mean", 
                          linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_xlim(0, self.window_size)
        ax.set_ylim(-3, 3)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Steps", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        def update(frame):
            """Update function for animation."""
            if frame < len(data_stream):
                data_point = data_stream[frame]
                self.data_buffer.append(data_point)
                self.stats_buffer.append(np.mean(self.data_buffer))
                
                # Update lines
                x_data = list(range(len(self.data_buffer)))
                line_data.set_data(x_data, list(self.data_buffer))
                line_ma.set_data(x_data, list(self.stats_buffer))
                
                # Update y-axis limits dynamically
                if len(self.data_buffer) > 0:
                    y_min = min(min(self.data_buffer), min(self.stats_buffer))
                    y_max = max(max(self.data_buffer), max(self.stats_buffer))
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim(y_min - margin, y_max + margin)
            
            return line_data, line_ma
        
        ani = animation.FuncAnimation(
            fig, update, frames=len(data_stream), 
            interval=self.update_interval, blit=True, repeat=False
        )
        
        return ani
    
    def show_animation(self, data_stream: np.ndarray, title: str = "Real-Time Analysis") -> None:
        """Show real-time animation.
        
        Args:
            data_stream: Stream of data points
            title: Animation title
        """
        ani = self.create_animation(data_stream, title)
        plt.show()
        return ani


class InteractiveVisualizer:
    """Interactive visualization using Plotly."""
    
    def __init__(self):
        """Initialize interactive visualizer."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required for interactive visualizations")
    
    def create_interactive_plot(
        self,
        series: pd.Series,
        title: str = "Interactive Time Series"
    ) -> go.Figure:
        """Create interactive time series plot.
        
        Args:
            series: Time series data
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name='Time Series',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            template="plotly_white"
        )
        
        return fig
    
    def create_forecast_plot(
        self,
        historical: pd.Series,
        forecast: pd.Series,
        confidence_intervals: Optional[Dict[str, pd.Series]] = None,
        title: str = "Interactive Forecast"
    ) -> go.Figure:
        """Create interactive forecast plot.
        
        Args:
            historical: Historical data
            forecast: Forecast data
            confidence_intervals: Optional confidence intervals
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Confidence intervals
        if confidence_intervals:
            for name, ci in confidence_intervals.items():
                fig.add_trace(go.Scatter(
                    x=ci.index,
                    y=ci.values,
                    mode='lines',
                    name=name,
                    line=dict(width=0),
                    showlegend=False
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            template="plotly_white"
        )
        
        return fig
