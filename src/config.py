"""Configuration management for time series analysis."""

import yaml
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data generation and preprocessing."""
    
    # Data generation
    length: int = 1000
    frequency: str = "1H"
    trend_strength: float = 0.1
    seasonality_strength: float = 0.5
    noise_level: float = 0.1
    anomaly_probability: float = 0.05
    seasonal_periods: int = 24
    
    # Preprocessing
    outlier_method: str = "iqr"
    outlier_factor: float = 1.5
    interpolation_method: str = "linear"
    normalization_method: str = "zscore"


@dataclass
class ModelConfig:
    """Configuration for time series models."""
    
    # ARIMA parameters
    arima_order: tuple = (1, 1, 1)
    seasonal_order: tuple = (0, 0, 0, 0)
    
    # Prophet parameters
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_daily_seasonality: bool = True
    
    # Anomaly detection parameters
    contamination: float = 0.1
    random_state: int = 42
    
    # Ensemble parameters
    ensemble_method: str = "average"


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    
    # Plot settings
    figure_size: tuple = (12, 6)
    style: str = "default"
    dpi: int = 300
    
    # Real-time settings
    window_size: int = 50
    update_interval: int = 100
    
    # Colors
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    anomaly_color: str = "#d62728"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """Main application configuration."""
    
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Application settings
    debug: bool = False
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    cache_dir: str = "cache"


class ConfigManager:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/config.yaml"
        self.config = AppConfig()
        
    def load_config(self, config_path: Optional[str] = None) -> AppConfig:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        path = config_path or self.config_path
        
        if not os.path.exists(path):
            logger.warning(f"Configuration file not found: {path}. Using default configuration.")
            return self.config
        
        try:
            with open(path, 'r') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    config_dict = yaml.safe_load(f)
                elif path.endswith('.json'):
                    config_dict = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {path}")
            
            # Convert to AppConfig
            self.config = self._dict_to_config(config_dict)
            logger.info(f"Configuration loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
        
        return self.config
    
    def save_config(self, config: AppConfig, config_path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
        """
        path = config_path or self.config_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            config_dict = asdict(config)
            
            with open(path, 'w') as f:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported configuration file format: {path}")
            
            logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            AppConfig object
        """
        try:
            # Convert nested dictionaries to dataclass objects
            data_config = DataConfig(**config_dict.get('data', {}))
            model_config = ModelConfig(**config_dict.get('model', {}))
            viz_config = VisualizationConfig(**config_dict.get('visualization', {}))
            log_config = LoggingConfig(**config_dict.get('logging', {}))
            
            # Create main config
            app_config = AppConfig(
                data=data_config,
                model=model_config,
                visualization=viz_config,
                logging=log_config,
                debug=config_dict.get('debug', False),
                data_dir=config_dict.get('data_dir', 'data'),
                models_dir=config_dict.get('models_dir', 'models'),
                logs_dir=config_dict.get('logs_dir', 'logs'),
                cache_dir=config_dict.get('cache_dir', 'cache')
            )
            
            return app_config
            
        except Exception as e:
            logger.error(f"Error converting configuration: {e}")
            return self.config
    
    def get_config(self) -> AppConfig:
        """Get current configuration.
        
        Returns:
            Current configuration
        """
        return self.config
    
    def update_config(self, **kwargs) -> None:
        """Update configuration with new values.
        
        Args:
            **kwargs: Configuration updates
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")


def setup_logging(config: LoggingConfig) -> None:
    """Setup logging configuration.
    
    Args:
        config: Logging configuration
    """
    # Create logs directory if it doesn't exist
    if config.file_path:
        os.makedirs(os.path.dirname(config.file_path), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        format=config.format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.file_path) if config.file_path else logging.NullHandler()
        ]
    )
    
    # Configure file handler with rotation if file path is provided
    if config.file_path:
        from logging.handlers import RotatingFileHandler
        
        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(logging.Formatter(config.format))
        
        # Add handler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def create_default_config() -> AppConfig:
    """Create default configuration.
    
    Returns:
        Default configuration
    """
    return AppConfig()


def load_config_from_env() -> AppConfig:
    """Load configuration from environment variables.
    
    Returns:
        Configuration from environment
    """
    config = create_default_config()
    
    # Load from environment variables
    env_mappings = {
        'TS_DEBUG': ('debug', bool),
        'TS_DATA_DIR': ('data_dir', str),
        'TS_MODELS_DIR': ('models_dir', str),
        'TS_LOGS_DIR': ('logs_dir', str),
        'TS_CACHE_DIR': ('cache_dir', str),
        'TS_LOG_LEVEL': ('logging.level', str),
        'TS_LOG_FILE': ('logging.file_path', str),
    }
    
    for env_var, (config_path, type_func) in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            try:
                if type_func == bool:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    value = type_func(value)
                
                # Set nested attribute
                if '.' in config_path:
                    parent, child = config_path.split('.', 1)
                    parent_obj = getattr(config, parent)
                    setattr(parent_obj, child, value)
                else:
                    setattr(config, config_path, value)
                    
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid environment variable {env_var}: {e}")
    
    return config
