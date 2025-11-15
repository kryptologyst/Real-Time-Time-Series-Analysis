"""Time series forecasting models and anomaly detection."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Import forecasting libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available. ARIMA models will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("prophet not available. Prophet models will be disabled.")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Anomaly detection will be disabled.")

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for time series models."""
    
    # ARIMA parameters
    arima_order: Tuple[int, int, int] = (1, 1, 1)
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    
    # Prophet parameters
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_daily_seasonality: bool = True
    
    # Anomaly detection parameters
    contamination: float = 0.1
    random_state: int = 42


class BaseForecaster(ABC):
    """Abstract base class for time series forecasters."""
    
    @abstractmethod
    def fit(self, series: pd.Series) -> None:
        """Fit the model to the time series data."""
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> pd.Series:
        """Generate predictions for future steps."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model."""
        pass


class ARIMAForecaster(BaseForecaster):
    """ARIMA-based time series forecaster."""
    
    def __init__(self, config: ModelConfig):
        """Initialize ARIMA forecaster.
        
        Args:
            config: Model configuration
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA forecasting")
        
        self.config = config
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
    def _check_stationarity(self, series: pd.Series) -> bool:
        """Check if series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to test
            
        Returns:
            True if stationary, False otherwise
        """
        result = adfuller(series.dropna())
        return result[1] < 0.05  # p-value < 0.05 indicates stationarity
    
    def _make_stationary(self, series: pd.Series) -> pd.Series:
        """Make series stationary by differencing if necessary.
        
        Args:
            series: Input time series
            
        Returns:
            Stationary series
        """
        current_series = series.copy()
        diff_count = 0
        
        while not self._check_stationarity(current_series) and diff_count < 2:
            current_series = current_series.diff().dropna()
            diff_count += 1
            
        return current_series
    
    def fit(self, series: pd.Series) -> None:
        """Fit ARIMA model to the time series.
        
        Args:
            series: Training time series data
        """
        try:
            # Make series stationary
            stationary_series = self._make_stationary(series)
            
            # Fit ARIMA model
            self.fitted_model = ARIMA(
                stationary_series,
                order=self.config.arima_order,
                seasonal_order=self.config.seasonal_order
            ).fit()
            
            self.is_fitted = True
            logger.info("ARIMA model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
    
    def predict(self, steps: int) -> pd.Series:
        """Generate ARIMA predictions.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            forecast = self.fitted_model.forecast(steps=steps)
            return pd.Series(forecast, name="arima_forecast")
        except Exception as e:
            logger.error(f"Error making ARIMA predictions: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ARIMA model information.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "model_type": "ARIMA",
            "order": self.config.arima_order,
            "seasonal_order": self.config.seasonal_order,
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
            "status": "fitted"
        }


class ProphetForecaster(BaseForecaster):
    """Prophet-based time series forecaster."""
    
    def __init__(self, config: ModelConfig):
        """Initialize Prophet forecaster.
        
        Args:
            config: Model configuration
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet forecasting")
        
        self.config = config
        self.model = None
        self.is_fitted = False
        
    def fit(self, series: pd.Series) -> None:
        """Fit Prophet model to the time series.
        
        Args:
            series: Training time series data
        """
        try:
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': series.index,
                'y': series.values
            })
            
            # Initialize and configure Prophet
            self.model = Prophet(
                yearly_seasonality=self.config.prophet_yearly_seasonality,
                weekly_seasonality=self.config.prophet_weekly_seasonality,
                daily_seasonality=self.config.prophet_daily_seasonality
            )
            
            # Fit the model
            self.model.fit(df)
            self.is_fitted = True
            logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {e}")
            raise
    
    def predict(self, steps: int) -> pd.Series:
        """Generate Prophet predictions.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=steps)
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Return only the future predictions
            future_forecast = forecast.tail(steps)
            return pd.Series(
                future_forecast['yhat'].values,
                index=future_forecast['ds'],
                name="prophet_forecast"
            )
            
        except Exception as e:
            logger.error(f"Error making Prophet predictions: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Prophet model information.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "model_type": "Prophet",
            "yearly_seasonality": self.config.prophet_yearly_seasonality,
            "weekly_seasonality": self.config.prophet_weekly_seasonality,
            "daily_seasonality": self.config.prophet_daily_seasonality,
            "status": "fitted"
        }


class AnomalyDetector:
    """Anomaly detection for time series data."""
    
    def __init__(self, config: ModelConfig):
        """Initialize anomaly detector.
        
        Args:
            config: Model configuration
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for anomaly detection")
        
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, series: pd.Series) -> None:
        """Fit anomaly detection model.
        
        Args:
            series: Training time series data
        """
        try:
            # Prepare features for anomaly detection
            features = self._create_features(series)
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Fit Isolation Forest
            self.model = IsolationForest(
                contamination=self.config.contamination,
                random_state=self.config.random_state
            )
            self.model.fit(scaled_features)
            
            self.is_fitted = True
            logger.info("Anomaly detection model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting anomaly detection model: {e}")
            raise
    
    def _create_features(self, series: pd.Series) -> np.ndarray:
        """Create features for anomaly detection.
        
        Args:
            series: Input time series
            
        Returns:
            Feature matrix
        """
        features = []
        
        # Rolling statistics
        for window in [5, 10, 20]:
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            features.extend([rolling_mean, rolling_std])
        
        # Lag features
        for lag in [1, 2, 3]:
            lag_feature = series.shift(lag)
            features.append(lag_feature)
        
        # Combine all features
        feature_df = pd.concat(features, axis=1)
        feature_df = feature_df.dropna()
        
        return feature_df.values
    
    def detect_anomalies(self, series: pd.Series) -> pd.Series:
        """Detect anomalies in time series.
        
        Args:
            series: Time series to analyze
            
        Returns:
            Boolean series indicating anomalies
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before detecting anomalies")
        
        try:
            # Create features
            features = self._create_features(series)
            
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Predict anomalies
            predictions = self.model.predict(scaled_features)
            
            # Convert to boolean (1 = normal, -1 = anomaly)
            anomalies = predictions == -1
            
            # Create result series with proper index
            result_index = series.index[len(series) - len(anomalies):]
            return pd.Series(anomalies, index=result_index, name="anomaly")
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get anomaly detection model information.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "model_type": "IsolationForest",
            "contamination": self.config.contamination,
            "status": "fitted"
        }


class ModelEnsemble:
    """Ensemble of multiple forecasting models."""
    
    def __init__(self, models: List[BaseForecaster]):
        """Initialize model ensemble.
        
        Args:
            models: List of forecasting models
        """
        self.models = models
        self.is_fitted = False
        
    def fit(self, series: pd.Series) -> None:
        """Fit all models in the ensemble.
        
        Args:
            series: Training time series data
        """
        for model in self.models:
            try:
                model.fit(series)
            except Exception as e:
                logger.warning(f"Failed to fit model {type(model).__name__}: {e}")
        
        self.is_fitted = True
        logger.info("Model ensemble fitted")
    
    def predict(self, steps: int, method: str = "average") -> pd.Series:
        """Generate ensemble predictions.
        
        Args:
            steps: Number of steps to predict
            method: Ensemble method ("average", "median")
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        
        for model in self.models:
            try:
                pred = model.predict(steps)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to get predictions from {type(model).__name__}: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions available")
        
        # Combine predictions
        if method == "average":
            ensemble_pred = pd.concat(predictions, axis=1).mean(axis=1)
        elif method == "median":
            ensemble_pred = pd.concat(predictions, axis=1).median(axis=1)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        return ensemble_pred
