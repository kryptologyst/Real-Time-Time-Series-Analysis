"""Data generation and preprocessing utilities for time series analysis."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesConfig:
    """Configuration for synthetic time series generation."""
    
    length: int = 1000
    frequency: str = "1H"
    trend_strength: float = 0.1
    seasonality_strength: float = 0.5
    noise_level: float = 0.1
    anomaly_probability: float = 0.05
    seasonal_periods: int = 24


class SyntheticDataGenerator:
    """Generate realistic synthetic time series data with various components."""
    
    def __init__(self, config: TimeSeriesConfig):
        """Initialize the data generator with configuration.
        
        Args:
            config: Configuration object for data generation
        """
        self.config = config
        
    def generate_time_index(self) -> pd.DatetimeIndex:
        """Generate time index for the series.
        
        Returns:
            DatetimeIndex with specified frequency and length
        """
        return pd.date_range(
            start="2023-01-01",
            periods=self.config.length,
            freq=self.config.frequency
        )
    
    def generate_trend(self, time_index: pd.DatetimeIndex) -> np.ndarray:
        """Generate linear trend component.
        
        Args:
            time_index: Time index for the series
            
        Returns:
            Array of trend values
        """
        trend = np.linspace(0, self.config.trend_strength * len(time_index), len(time_index))
        return trend
    
    def generate_seasonality(self, time_index: pd.DatetimeIndex) -> np.ndarray:
        """Generate seasonal component.
        
        Args:
            time_index: Time index for the series
            
        Returns:
            Array of seasonal values
        """
        # Daily seasonality
        daily_seasonal = self.config.seasonality_strength * np.sin(
            2 * np.pi * np.arange(len(time_index)) / self.config.seasonal_periods
        )
        
        # Weekly seasonality
        weekly_seasonal = 0.3 * self.config.seasonality_strength * np.sin(
            2 * np.pi * np.arange(len(time_index)) / (self.config.seasonal_periods * 7)
        )
        
        return daily_seasonal + weekly_seasonal
    
    def generate_noise(self, length: int) -> np.ndarray:
        """Generate random noise component.
        
        Args:
            length: Length of the noise array
            
        Returns:
            Array of noise values
        """
        return self.config.noise_level * np.random.randn(length)
    
    def generate_anomalies(self, time_index: pd.DatetimeIndex) -> np.ndarray:
        """Generate anomaly points.
        
        Args:
            time_index: Time index for the series
            
        Returns:
            Array of anomaly values (0 for normal, non-zero for anomalies)
        """
        anomalies = np.zeros(len(time_index))
        num_anomalies = int(self.config.anomaly_probability * len(time_index))
        
        if num_anomalies > 0:
            anomaly_indices = np.random.choice(
                len(time_index), 
                size=num_anomalies, 
                replace=False
            )
            anomalies[anomaly_indices] = np.random.choice(
                [-3, 3], 
                size=num_anomalies
            ) * self.config.noise_level
        
        return anomalies
    
    def generate_series(self, seed: Optional[int] = None) -> pd.Series:
        """Generate complete synthetic time series.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Complete time series with all components
        """
        if seed is not None:
            np.random.seed(seed)
            
        time_index = self.generate_time_index()
        trend = self.generate_trend(time_index)
        seasonality = self.generate_seasonality(time_index)
        noise = self.generate_noise(len(time_index))
        anomalies = self.generate_anomalies(time_index)
        
        # Combine all components
        series = trend + seasonality + noise + anomalies
        
        return pd.Series(series, index=time_index, name="synthetic_series")


class DataPreprocessor:
    """Preprocessing utilities for time series data."""
    
    @staticmethod
    def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
        """Detect outliers using IQR method.
        
        Args:
            series: Input time series
            factor: IQR factor for outlier detection
            
        Returns:
            Boolean series indicating outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    @staticmethod
    def remove_outliers(series: pd.Series, method: str = "iqr", **kwargs) -> pd.Series:
        """Remove outliers from time series.
        
        Args:
            series: Input time series
            method: Method for outlier detection ("iqr", "zscore")
            **kwargs: Additional arguments for outlier detection
            
        Returns:
            Series with outliers replaced by NaN
        """
        if method == "iqr":
            outliers = DataPreprocessor.detect_outliers_iqr(series, **kwargs)
        elif method == "zscore":
            threshold = kwargs.get("threshold", 3)
            outliers = np.abs(series - series.mean()) > threshold * series.std()
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        cleaned_series = series.copy()
        cleaned_series[outliers] = np.nan
        
        return cleaned_series
    
    @staticmethod
    def interpolate_missing(series: pd.Series, method: str = "linear") -> pd.Series:
        """Interpolate missing values in time series.
        
        Args:
            series: Input time series with missing values
            method: Interpolation method
            
        Returns:
            Series with interpolated values
        """
        return series.interpolate(method=method)
    
    @staticmethod
    def normalize_series(series: pd.Series, method: str = "zscore") -> pd.Series:
        """Normalize time series data.
        
        Args:
            series: Input time series
            method: Normalization method ("zscore", "minmax", "robust")
            
        Returns:
            Normalized series
        """
        if method == "zscore":
            return (series - series.mean()) / series.std()
        elif method == "minmax":
            return (series - series.min()) / (series.max() - series.min())
        elif method == "robust":
            median = series.median()
            mad = np.median(np.abs(series - median))
            return (series - median) / mad
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class RollingWindowAnalyzer:
    """Real-time rolling window analysis for time series."""
    
    def __init__(self, window_size: int = 30):
        """Initialize rolling window analyzer.
        
        Args:
            window_size: Size of the rolling window
        """
        self.window_size = window_size
        self.data_buffer = []
        self.stats_buffer = {}
        
    def add_data_point(self, value: float) -> Dict[str, float]:
        """Add new data point and compute rolling statistics.
        
        Args:
            value: New data point value
            
        Returns:
            Dictionary of computed statistics
        """
        self.data_buffer.append(value)
        
        # Keep only the last window_size points
        if len(self.data_buffer) > self.window_size:
            self.data_buffer = self.data_buffer[-self.window_size:]
        
        # Compute statistics
        stats = {
            "mean": np.mean(self.data_buffer),
            "std": np.std(self.data_buffer),
            "min": np.min(self.data_buffer),
            "max": np.max(self.data_buffer),
            "median": np.median(self.data_buffer),
            "current": value,
            "window_size": len(self.data_buffer)
        }
        
        self.stats_buffer = stats
        return stats
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current rolling statistics.
        
        Returns:
            Dictionary of current statistics
        """
        return self.stats_buffer.copy()
    
    def reset(self) -> None:
        """Reset the analyzer buffers."""
        self.data_buffer = []
        self.stats_buffer = {}
