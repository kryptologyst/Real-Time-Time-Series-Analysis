# Real-Time Time Series Analysis

A comprehensive Python package for real-time time series analysis, featuring state-of-the-art forecasting methods, anomaly detection, and interactive visualization capabilities.

## Features

- **Synthetic Data Generation**: Create realistic time series with trends, seasonality, noise, and anomalies
- **Advanced Forecasting**: ARIMA, Prophet, and ensemble methods for accurate predictions
- **Anomaly Detection**: Isolation Forest-based anomaly detection with configurable parameters
- **Real-Time Analysis**: Rolling window statistics and live visualization
- **Interactive Web Interface**: Streamlit-based dashboard for exploring results
- **Comprehensive Testing**: Unit tests for all major components
- **Configuration Management**: YAML/JSON configuration with environment variable support
- **Checkpoint System**: Save and load models, data, and experiment results
- **Modern Python**: Type hints, docstrings, and PEP8 compliance

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies

For advanced features, you can install additional packages:

```bash
# Deep learning models
pip install tensorflow keras

# Additional forecasting libraries
pip install darts sktime tslearn

# Gradient boosting
pip install xgboost lightgbm catboost
```

## Quick Start

### 1. Generate Synthetic Data

```python
from src.data_utils import SyntheticDataGenerator, TimeSeriesConfig

# Configure data generation
config = TimeSeriesConfig(
    length=1000,
    trend_strength=0.1,
    seasonality_strength=0.5,
    noise_level=0.1,
    anomaly_probability=0.05
)

# Generate data
generator = SyntheticDataGenerator(config)
series = generator.generate_series(seed=42)
```

### 2. Run Forecasting Models

```python
from src.models import ARIMAForecaster, ProphetForecaster, ModelConfig

# Configure models
model_config = ModelConfig()

# ARIMA forecasting
arima_model = ARIMAForecaster(model_config)
arima_model.fit(series)
arima_forecast = arima_model.predict(steps=30)

# Prophet forecasting
prophet_model = ProphetForecaster(model_config)
prophet_model.fit(series)
prophet_forecast = prophet_model.predict(steps=30)
```

### 3. Detect Anomalies

```python
from src.models import AnomalyDetector

# Configure anomaly detection
detector = AnomalyDetector(model_config)
detector.fit(series)
anomalies = detector.detect_anomalies(series)
```

### 4. Launch Web Interface

```bash
streamlit run app.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_utils.py      # Data generation and preprocessing
│   ├── models.py          # Forecasting and anomaly detection models
│   ├── visualization.py   # Plotting and visualization utilities
│   ├── config.py         # Configuration management
│   └── checkpoint.py     # Model and data checkpointing
├── tests/                 # Unit tests
│   └── test_timeseries.py
├── notebooks/             # Jupyter notebooks for exploration
├── data/                  # Data storage directory
├── models/                # Saved models directory
├── logs/                  # Log files directory
├── config/                # Configuration files
├── app.py                 # Streamlit web application
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Usage Examples

### Basic Time Series Analysis

```python
import pandas as pd
import numpy as np
from src.data_utils import SyntheticDataGenerator, TimeSeriesConfig
from src.models import ARIMAForecaster, AnomalyDetector
from src.visualization import TimeSeriesVisualizer

# Generate synthetic data
config = TimeSeriesConfig(length=500)
generator = SyntheticDataGenerator(config)
series = generator.generate_series(seed=42)

# Fit ARIMA model
arima_model = ARIMAForecaster()
arima_model.fit(series)
forecast = arima_model.predict(steps=50)

# Detect anomalies
detector = AnomalyDetector()
detector.fit(series)
anomalies = detector.detect_anomalies(series)

# Visualize results
visualizer = TimeSeriesVisualizer()
visualizer.plot_forecast(series, forecast)
visualizer.plot_anomalies(series, anomalies)
```

### Real-Time Analysis

```python
from src.data_utils import RollingWindowAnalyzer
from src.visualization import RealTimeVisualizer

# Initialize rolling window analyzer
analyzer = RollingWindowAnalyzer(window_size=30)

# Process data stream
for value in series:
    stats = analyzer.add_data_point(value)
    print(f"Current mean: {stats['mean']:.3f}, std: {stats['std']:.3f}")

# Create real-time visualization
rt_visualizer = RealTimeVisualizer()
rt_visualizer.show_animation(series.values)
```

### Configuration Management

```python
from src.config import ConfigManager, AppConfig

# Load configuration
config_manager = ConfigManager("config/config.yaml")
config = config_manager.load_config()

# Update configuration
config_manager.update_config(debug=True)

# Save configuration
config_manager.save_config(config)
```

### Model Checkpointing

```python
from src.checkpoint import ModelCheckpoint, DataCheckpoint

# Save model
checkpoint_manager = ModelCheckpoint("models/checkpoints")
checkpoint_path = checkpoint_manager.save_model(
    arima_model, "arima_model", {"accuracy": 0.95}
)

# Load model
loaded_model, metadata = checkpoint_manager.load_model(checkpoint_path)

# Save data
data_checkpoint = DataCheckpoint("data/checkpoints")
data_path = data_checkpoint.save_data(series, "training_data")
```

## Web Interface

The Streamlit web interface provides an interactive dashboard for:

- **Data Overview**: Generate and explore synthetic time series
- **Forecasting**: Run multiple forecasting models and compare results
- **Anomaly Detection**: Detect and visualize anomalies in the data
- **Real-Time Analysis**: Monitor rolling window statistics
- **Interactive Plots**: Explore data with interactive visualizations

### Launch the Web Interface

```bash
streamlit run app.py
```

The interface will be available at `http://localhost:8501`

## Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_timeseries.py -v
```

## Configuration

The application supports configuration through:

1. **YAML/JSON files**: `config/config.yaml`
2. **Environment variables**: `TS_DEBUG=true`, `TS_LOG_LEVEL=INFO`
3. **Code configuration**: Direct instantiation of config objects

### Example Configuration File

```yaml
data:
  length: 1000
  trend_strength: 0.1
  seasonality_strength: 0.5
  noise_level: 0.1
  anomaly_probability: 0.05

model:
  arima_order: [1, 1, 1]
  seasonal_order: [0, 0, 0, 0]
  contamination: 0.1
  random_state: 42

visualization:
  figure_size: [12, 6]
  style: "default"
  dpi: 300

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: "logs/app.log"
```

## Advanced Features

### Ensemble Forecasting

```python
from src.models import ModelEnsemble

# Create ensemble of multiple models
models = [ARIMAForecaster(), ProphetForecaster()]
ensemble = ModelEnsemble(models)
ensemble.fit(series)
ensemble_forecast = ensemble.predict(steps=30, method="average")
```

### Experiment Tracking

```python
from src.checkpoint import ExperimentTracker

# Track experiments
tracker = ExperimentTracker("experiments")
experiment_id = tracker.start_experiment(
    "forecasting_experiment",
    "Compare ARIMA vs Prophet",
    {"arima_order": (1,1,1), "prophet_seasonality": True}
)

# Log metrics
tracker.log_metric(experiment_id, "mae", 0.05, step=1)
tracker.log_metric(experiment_id, "rmse", 0.08, step=1)

# Save results
tracker.save_result(experiment_id, "forecast", forecast_data)
tracker.finish_experiment(experiment_id, {"final_mae": 0.05})
```

## Performance Considerations

- **Memory Usage**: Large datasets may require chunked processing
- **Model Training**: ARIMA and Prophet models can be computationally intensive
- **Real-Time Processing**: Consider using smaller window sizes for live analysis
- **Visualization**: Interactive plots may be slow with very large datasets

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Model Fitting Failures**: Check data quality and model parameters
3. **Memory Issues**: Reduce data size or use chunked processing
4. **Visualization Problems**: Ensure matplotlib backend is properly configured

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or set environment variable:

```bash
export TS_DEBUG=true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with modern Python libraries (pandas, numpy, matplotlib, plotly)
- Forecasting models from statsmodels and Prophet
- Anomaly detection using scikit-learn
- Web interface powered by Streamlit
- Testing framework with pytest

## Future Enhancements

- [ ] Deep learning models (LSTM, GRU, Transformers)
- [ ] Probabilistic forecasting
- [ ] Causal inference methods
- [ ] Real-time data streaming integration
- [ ] Advanced visualization dashboards
- [ ] Model deployment and serving
- [ ] Distributed computing support
# Real-Time-Time-Series-Analysis
