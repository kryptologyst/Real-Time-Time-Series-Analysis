#!/usr/bin/env python3
"""Command-line interface for time series analysis."""

import argparse
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_utils import SyntheticDataGenerator, TimeSeriesConfig
from models import ARIMAForecaster, ProphetForecaster, AnomalyDetector, ModelConfig
from visualization import TimeSeriesVisualizer
from config import ConfigManager


def generate_data(args):
    """Generate synthetic time series data."""
    print("Generating synthetic time series data...")
    
    config = TimeSeriesConfig(
        length=args.length,
        trend_strength=args.trend_strength,
        seasonality_strength=args.seasonality_strength,
        noise_level=args.noise_level,
        anomaly_probability=args.anomaly_probability
    )
    
    generator = SyntheticDataGenerator(config)
    series = generator.generate_series(seed=args.seed)
    
    # Save data
    output_path = Path(args.output) / "synthetic_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(output_path)
    
    print(f"Data saved to {output_path}")
    print(f"Generated {len(series)} data points")
    print(f"Mean: {series.mean():.3f}, Std: {series.std():.3f}")
    
    return series


def run_forecasting(args):
    """Run forecasting models."""
    print("Running forecasting models...")
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found")
        return
    
    series = pd.read_csv(data_path, index_col=0, parse_dates=True).squeeze()
    
    # Split data
    split_point = int(len(series) * 0.8)
    train_series = series[:split_point]
    test_series = series[split_point:]
    
    # Configure models
    model_config = ModelConfig()
    
    results = {}
    
    # ARIMA
    if args.arima:
        print("Fitting ARIMA model...")
        try:
            arima_model = ARIMAForecaster(model_config)
            arima_model.fit(train_series)
            arima_forecast = arima_model.predict(len(test_series))
            results['ARIMA'] = arima_forecast
            print("ARIMA model completed successfully")
        except Exception as e:
            print(f"ARIMA model failed: {e}")
    
    # Prophet
    if args.prophet:
        print("Fitting Prophet model...")
        try:
            prophet_model = ProphetForecaster(model_config)
            prophet_model.fit(train_series)
            prophet_forecast = prophet_model.predict(len(test_series))
            results['Prophet'] = prophet_forecast
            print("Prophet model completed successfully")
        except Exception as e:
            print(f"Prophet model failed: {e}")
    
    # Save results
    if results:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, forecast in results.items():
            output_path = output_dir / f"{model_name.lower()}_forecast.csv"
            forecast.to_csv(output_path)
            print(f"{model_name} forecast saved to {output_path}")
    
    return results


def detect_anomalies(args):
    """Detect anomalies in time series data."""
    print("Detecting anomalies...")
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found")
        return
    
    series = pd.read_csv(data_path, index_col=0, parse_dates=True).squeeze()
    
    # Configure detector
    model_config = ModelConfig(contamination=args.contamination)
    detector = AnomalyDetector(model_config)
    
    # Fit and detect
    detector.fit(series)
    anomalies = detector.detect_anomalies(series)
    
    # Save results
    output_path = Path(args.output) / "anomalies.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anomalies.to_csv(output_path)
    
    print(f"Anomaly detection completed")
    print(f"Detected {anomalies.sum()} anomalies out of {len(series)} data points")
    print(f"Anomaly rate: {anomalies.sum() / len(series) * 100:.2f}%")
    print(f"Results saved to {output_path}")
    
    return anomalies


def visualize(args):
    """Create visualizations."""
    print("Creating visualizations...")
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file {data_path} not found")
        return
    
    series = pd.read_csv(data_path, index_col=0, parse_dates=True).squeeze()
    
    # Create visualizer
    visualizer = TimeSeriesVisualizer()
    
    # Create plots
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Time series plot
    fig = visualizer.plot_time_series(series, "Time Series Analysis")
    fig.savefig(output_dir / "time_series.png", dpi=300, bbox_inches='tight')
    print(f"Time series plot saved to {output_dir / 'time_series.png'}")
    
    # Decomposition plot
    try:
        fig = visualizer.plot_decomposition(series)
        fig.savefig(output_dir / "decomposition.png", dpi=300, bbox_inches='tight')
        print(f"Decomposition plot saved to {output_dir / 'decomposition.png'}")
    except Exception as e:
        print(f"Decomposition plot failed: {e}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Time Series Analysis CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate data command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--length', type=int, default=1000, help='Data length')
    gen_parser.add_argument('--trend-strength', type=float, default=0.1, help='Trend strength')
    gen_parser.add_argument('--seasonality-strength', type=float, default=0.5, help='Seasonality strength')
    gen_parser.add_argument('--noise-level', type=float, default=0.1, help='Noise level')
    gen_parser.add_argument('--anomaly-probability', type=float, default=0.05, help='Anomaly probability')
    gen_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    gen_parser.add_argument('--output', type=str, default='data', help='Output directory')
    
    # Forecasting command
    forecast_parser = subparsers.add_parser('forecast', help='Run forecasting models')
    forecast_parser.add_argument('--data', type=str, required=True, help='Input data file')
    forecast_parser.add_argument('--arima', action='store_true', help='Run ARIMA model')
    forecast_parser.add_argument('--prophet', action='store_true', help='Run Prophet model')
    forecast_parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    # Anomaly detection command
    anomaly_parser = subparsers.add_parser('anomaly', help='Detect anomalies')
    anomaly_parser.add_argument('--data', type=str, required=True, help='Input data file')
    anomaly_parser.add_argument('--contamination', type=float, default=0.1, help='Contamination rate')
    anomaly_parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
    viz_parser.add_argument('--data', type=str, required=True, help='Input data file')
    viz_parser.add_argument('--output', type=str, default='plots', help='Output directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Import pandas here to avoid import issues
    import pandas as pd
    
    try:
        if args.command == 'generate':
            generate_data(args)
        elif args.command == 'forecast':
            run_forecasting(args)
        elif args.command == 'anomaly':
            detect_anomalies(args)
        elif args.command == 'visualize':
            visualize(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
