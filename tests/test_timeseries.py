"""Unit tests for time series analysis package."""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_utils import (
    SyntheticDataGenerator, 
    TimeSeriesConfig, 
    DataPreprocessor, 
    RollingWindowAnalyzer
)
from models import (
    ARIMAForecaster, 
    ProphetForecaster, 
    AnomalyDetector, 
    ModelEnsemble, 
    ModelConfig
)
from config import ConfigManager, AppConfig
from checkpoint import ModelCheckpoint, DataCheckpoint, ExperimentTracker


class TestSyntheticDataGenerator(unittest.TestCase):
    """Test cases for SyntheticDataGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TimeSeriesConfig(
            length=100,
            trend_strength=0.1,
            seasonality_strength=0.5,
            noise_level=0.1,
            anomaly_probability=0.05
        )
        self.generator = SyntheticDataGenerator(self.config)
    
    def test_generate_time_index(self):
        """Test time index generation."""
        time_index = self.generator.generate_time_index()
        
        self.assertEqual(len(time_index), self.config.length)
        self.assertIsInstance(time_index, pd.DatetimeIndex)
        self.assertEqual(time_index.freq, self.config.frequency)
    
    def test_generate_trend(self):
        """Test trend generation."""
        time_index = self.generator.generate_time_index()
        trend = self.generator.generate_trend(time_index)
        
        self.assertEqual(len(trend), len(time_index))
        self.assertIsInstance(trend, np.ndarray)
        self.assertGreater(trend[-1], trend[0])  # Positive trend
    
    def test_generate_seasonality(self):
        """Test seasonality generation."""
        time_index = self.generator.generate_time_index()
        seasonality = self.generator.generate_seasonality(time_index)
        
        self.assertEqual(len(seasonality), len(time_index))
        self.assertIsInstance(seasonality, np.ndarray)
    
    def test_generate_noise(self):
        """Test noise generation."""
        length = 100
        noise = self.generator.generate_noise(length)
        
        self.assertEqual(len(noise), length)
        self.assertIsInstance(noise, np.ndarray)
        self.assertAlmostEqual(np.mean(noise), 0, delta=0.1)
    
    def test_generate_anomalies(self):
        """Test anomaly generation."""
        time_index = self.generator.generate_time_index()
        anomalies = self.generator.generate_anomalies(time_index)
        
        self.assertEqual(len(anomalies), len(time_index))
        self.assertIsInstance(anomalies, np.ndarray)
        self.assertGreaterEqual(np.sum(anomalies != 0), 0)
    
    def test_generate_series(self):
        """Test complete series generation."""
        series = self.generator.generate_series(seed=42)
        
        self.assertEqual(len(series), self.config.length)
        self.assertIsInstance(series, pd.Series)
        self.assertIsInstance(series.index, pd.DatetimeIndex)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test series with outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 90)
        outliers = np.array([10, -10, 15, -15])  # Clear outliers
        self.test_series = pd.Series(np.concatenate([normal_data, outliers]))
    
    def test_detect_outliers_iqr(self):
        """Test IQR outlier detection."""
        outliers = DataPreprocessor.detect_outliers_iqr(self.test_series)
        
        self.assertIsInstance(outliers, pd.Series)
        self.assertEqual(len(outliers), len(self.test_series))
        self.assertTrue(outliers.dtype == bool)
    
    def test_remove_outliers_iqr(self):
        """Test outlier removal with IQR method."""
        cleaned = DataPreprocessor.remove_outliers(self.test_series, method="iqr")
        
        self.assertIsInstance(cleaned, pd.Series)
        self.assertEqual(len(cleaned), len(self.test_series))
        self.assertTrue(cleaned.isna().any())  # Should have NaN values
    
    def test_remove_outliers_zscore(self):
        """Test outlier removal with Z-score method."""
        cleaned = DataPreprocessor.remove_outliers(self.test_series, method="zscore", threshold=2)
        
        self.assertIsInstance(cleaned, pd.Series)
        self.assertEqual(len(cleaned), len(self.test_series))
    
    def test_interpolate_missing(self):
        """Test missing value interpolation."""
        series_with_nans = self.test_series.copy()
        series_with_nans.iloc[10:15] = np.nan
        
        interpolated = DataPreprocessor.interpolate_missing(series_with_nans)
        
        self.assertIsInstance(interpolated, pd.Series)
        self.assertFalse(interpolated.isna().any())
    
    def test_normalize_series_zscore(self):
        """Test Z-score normalization."""
        normalized = DataPreprocessor.normalize_series(self.test_series, method="zscore")
        
        self.assertIsInstance(normalized, pd.Series)
        self.assertAlmostEqual(normalized.mean(), 0, delta=1e-10)
        self.assertAlmostEqual(normalized.std(), 1, delta=1e-10)
    
    def test_normalize_series_minmax(self):
        """Test min-max normalization."""
        normalized = DataPreprocessor.normalize_series(self.test_series, method="minmax")
        
        self.assertIsInstance(normalized, pd.Series)
        self.assertAlmostEqual(normalized.min(), 0, delta=1e-10)
        self.assertAlmostEqual(normalized.max(), 1, delta=1e-10)


class TestRollingWindowAnalyzer(unittest.TestCase):
    """Test cases for RollingWindowAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RollingWindowAnalyzer(window_size=10)
    
    def test_add_data_point(self):
        """Test adding data points and computing statistics."""
        test_values = [1, 2, 3, 4, 5]
        
        for value in test_values:
            stats = self.analyzer.add_data_point(value)
            
            self.assertIsInstance(stats, dict)
            self.assertIn("mean", stats)
            self.assertIn("std", stats)
            self.assertIn("current", stats)
            self.assertEqual(stats["current"], value)
    
    def test_window_size_limit(self):
        """Test that window size is maintained."""
        # Add more points than window size
        for i in range(20):
            self.analyzer.add_data_point(i)
        
        stats = self.analyzer.get_current_stats()
        self.assertEqual(stats["window_size"], 10)
    
    def test_reset(self):
        """Test resetting the analyzer."""
        self.analyzer.add_data_point(1)
        self.analyzer.reset()
        
        stats = self.analyzer.get_current_stats()
        self.assertEqual(len(stats), 0)


class TestModelConfig(unittest.TestCase):
    """Test cases for ModelConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ModelConfig()
        
        self.assertIsInstance(config.arima_order, tuple)
        self.assertIsInstance(config.seasonal_order, tuple)
        self.assertIsInstance(config.contamination, float)
        self.assertIsInstance(config.random_state, int)


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.yaml")
        self.config_manager = ConfigManager(self.config_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = self.config_manager.get_config()
        
        self.assertIsInstance(config, AppConfig)
        self.assertIsInstance(config.data, type(self.config_manager.config.data))
        self.assertIsInstance(config.model, type(self.config_manager.config.model))
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        config = AppConfig()
        config.debug = True
        
        # Save config
        self.config_manager.save_config(config)
        
        # Load config
        loaded_config = self.config_manager.load_config()
        
        self.assertTrue(loaded_config.debug)
    
    def test_update_config(self):
        """Test configuration updates."""
        self.config_manager.update_config(debug=True)
        
        config = self.config_manager.get_config()
        self.assertTrue(config.debug)


class TestModelCheckpoint(unittest.TestCase):
    """Test cases for ModelCheckpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = ModelCheckpoint(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_model(self):
        """Test saving and loading models."""
        # Create a simple model-like object
        model = {"weights": [1, 2, 3], "bias": 0.5}
        metadata = {"model_type": "test", "accuracy": 0.95}
        
        # Save model
        checkpoint_path = self.checkpoint_manager.save_model(
            model, "test_model", metadata
        )
        
        # Load model
        loaded_model, loaded_metadata = self.checkpoint_manager.load_model(checkpoint_path)
        
        self.assertEqual(loaded_model, model)
        self.assertEqual(loaded_metadata["model_type"], "test")
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        # Create multiple checkpoints
        model1 = {"weights": [1, 2, 3]}
        model2 = {"weights": [4, 5, 6]}
        
        self.checkpoint_manager.save_model(model1, "model1")
        self.checkpoint_manager.save_model(model2, "model2")
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        self.assertEqual(len(checkpoints), 2)
        self.assertTrue(any(cp["name"] == "model1" for cp in checkpoints))
        self.assertTrue(any(cp["name"] == "model2" for cp in checkpoints))


class TestDataCheckpoint(unittest.TestCase):
    """Test cases for DataCheckpoint."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_checkpoint = DataCheckpoint(self.temp_dir)
        
        # Create test data
        self.test_series = pd.Series(
            np.random.randn(100),
            index=pd.date_range("2023-01-01", periods=100, freq="H")
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_data_parquet(self):
        """Test saving and loading data in parquet format."""
        metadata = {"source": "test", "description": "test data"}
        
        # Save data
        checkpoint_path = self.data_checkpoint.save_data(
            self.test_series, "test_data", metadata, format="parquet"
        )
        
        # Load data
        loaded_data, loaded_metadata = self.data_checkpoint.load_data(checkpoint_path)
        
        pd.testing.assert_series_equal(loaded_data, self.test_series)
        self.assertEqual(loaded_metadata["source"], "test")
    
    def test_save_and_load_data_csv(self):
        """Test saving and loading data in CSV format."""
        # Save data
        checkpoint_path = self.data_checkpoint.save_data(
            self.test_series, "test_data", format="csv"
        )
        
        # Load data
        loaded_data, loaded_metadata = self.data_checkpoint.load_data(checkpoint_path)
        
        pd.testing.assert_series_equal(loaded_data, self.test_series)


class TestExperimentTracker(unittest.TestCase):
    """Test cases for ExperimentTracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = ExperimentTracker(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_start_experiment(self):
        """Test starting an experiment."""
        experiment_id = self.tracker.start_experiment(
            "test_experiment",
            "Test experiment description",
            {"param1": 1, "param2": "test"}
        )
        
        self.assertIsInstance(experiment_id, str)
        self.assertTrue(experiment_id.startswith("test_experiment"))
    
    def test_log_metric(self):
        """Test logging metrics."""
        experiment_id = self.tracker.start_experiment("test_experiment")
        
        self.tracker.log_metric(experiment_id, "accuracy", 0.95, step=1)
        self.tracker.log_metric(experiment_id, "loss", 0.1, step=1)
        
        # This should not raise an exception
        self.assertTrue(True)
    
    def test_save_result(self):
        """Test saving experiment results."""
        experiment_id = self.tracker.start_experiment("test_experiment")
        
        result_data = {"accuracy": 0.95, "loss": 0.1}
        self.tracker.save_result(experiment_id, "final_results", result_data)
        
        # This should not raise an exception
        self.assertTrue(True)
    
    def test_finish_experiment(self):
        """Test finishing an experiment."""
        experiment_id = self.tracker.start_experiment("test_experiment")
        
        final_results = {"final_accuracy": 0.95}
        self.tracker.finish_experiment(experiment_id, final_results)
        
        # This should not raise an exception
        self.assertTrue(True)
    
    def test_list_experiments(self):
        """Test listing experiments."""
        experiment_id1 = self.tracker.start_experiment("experiment1")
        experiment_id2 = self.tracker.start_experiment("experiment2")
        
        experiments = self.tracker.list_experiments()
        
        self.assertEqual(len(experiments), 2)
        self.assertTrue(any(exp["experiment_name"] == "experiment1" for exp in experiments))
        self.assertTrue(any(exp["experiment_name"] == "experiment2" for exp in experiments))


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Generate data
        config = TimeSeriesConfig(length=100)
        generator = SyntheticDataGenerator(config)
        series = generator.generate_series(seed=42)
        
        # Preprocess data
        cleaned_series = DataPreprocessor.remove_outliers(series)
        interpolated_series = DataPreprocessor.interpolate_missing(cleaned_series)
        normalized_series = DataPreprocessor.normalize_series(interpolated_series)
        
        # Test rolling window analysis
        analyzer = RollingWindowAnalyzer(window_size=10)
        for value in normalized_series.head(20):
            stats = analyzer.add_data_point(value)
            self.assertIsInstance(stats, dict)
        
        # This should complete without errors
        self.assertTrue(True)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestSyntheticDataGenerator,
        TestDataPreprocessor,
        TestRollingWindowAnalyzer,
        TestModelConfig,
        TestConfigManager,
        TestModelCheckpoint,
        TestDataCheckpoint,
        TestExperimentTracker,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
