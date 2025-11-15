"""Checkpoint saving and loading utilities for time series models."""

import pickle
import json
import os
from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """Model checkpoint manager for saving and loading models."""
    
    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ) -> str:
        """Save model to checkpoint.
        
        Args:
            model: Model object to save
            model_name: Name of the model
            metadata: Optional metadata to save with model
            version: Optional version string
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = version or timestamp
        
        checkpoint_name = f"{model_name}_{version}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        try:
            # Save model
            model_path = checkpoint_path / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            metadata = metadata or {}
            metadata.update({
                "model_name": model_name,
                "version": version,
                "timestamp": timestamp,
                "checkpoint_path": str(checkpoint_path)
            })
            
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved to {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, checkpoint_path: str) -> tuple[Any, Dict[str, Any]]:
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Tuple of (model, metadata)
        """
        checkpoint_path = Path(checkpoint_path)
        
        try:
            # Load model
            model_path = checkpoint_path / "model.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata_path = checkpoint_path / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"Model loaded from {checkpoint_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def list_checkpoints(self, model_name: Optional[str] = None) -> list[Dict[str, Any]]:
        """List available checkpoints.
        
        Args:
            model_name: Optional model name filter
            
        Returns:
            List of checkpoint information
        """
        checkpoints = []
        
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue
            
            metadata_path = checkpoint_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                if model_name is None or metadata.get("model_name") == model_name:
                    checkpoints.append({
                        "path": str(checkpoint_dir),
                        "name": metadata.get("model_name", checkpoint_dir.name),
                        "version": metadata.get("version", "unknown"),
                        "timestamp": metadata.get("timestamp", "unknown")
                    })
            except Exception as e:
                logger.warning(f"Error reading checkpoint metadata {checkpoint_dir}: {e}")
        
        return sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """Delete checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import shutil
            shutil.rmtree(checkpoint_path)
            logger.info(f"Checkpoint deleted: {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting checkpoint: {e}")
            return False


class DataCheckpoint:
    """Data checkpoint manager for saving and loading datasets."""
    
    def __init__(self, data_dir: str = "data/checkpoints"):
        """Initialize data checkpoint manager.
        
        Args:
            data_dir: Directory to store data checkpoints
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_data(
        self,
        data: Union[pd.DataFrame, pd.Series],
        data_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        format: str = "parquet"
    ) -> str:
        """Save data to checkpoint.
        
        Args:
            data: Data to save
            data_name: Name of the dataset
            metadata: Optional metadata
            format: Save format ("parquet", "csv", "pickle")
            
        Returns:
            Path to saved data
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{data_name}_{timestamp}"
        checkpoint_path = self.data_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        try:
            # Save data
            if format == "parquet":
                data_path = checkpoint_path / "data.parquet"
                data.to_parquet(data_path)
            elif format == "csv":
                data_path = checkpoint_path / "data.csv"
                data.to_csv(data_path)
            elif format == "pickle":
                data_path = checkpoint_path / "data.pkl"
                data.to_pickle(data_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Save metadata
            metadata = metadata or {}
            metadata.update({
                "data_name": data_name,
                "timestamp": timestamp,
                "format": format,
                "shape": data.shape if hasattr(data, 'shape') else len(data),
                "dtype": str(data.dtype) if hasattr(data, 'dtype') else str(type(data))
            })
            
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Data saved to {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def load_data(self, checkpoint_path: str) -> tuple[Union[pd.DataFrame, pd.Series], Dict[str, Any]]:
        """Load data from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Tuple of (data, metadata)
        """
        checkpoint_path = Path(checkpoint_path)
        
        try:
            # Load metadata first
            metadata_path = checkpoint_path / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Load data based on format
            format_type = metadata.get("format", "parquet")
            
            if format_type == "parquet":
                data_path = checkpoint_path / "data.parquet"
                data = pd.read_parquet(data_path)
            elif format_type == "csv":
                data_path = checkpoint_path / "data.csv"
                data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            elif format_type == "pickle":
                data_path = checkpoint_path / "data.pkl"
                data = pd.read_pickle(data_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Data loaded from {checkpoint_path}")
            return data, metadata
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise


class ExperimentTracker:
    """Track experiments and their results."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        """Initialize experiment tracker.
        
        Args:
            experiments_dir: Directory to store experiment data
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
    
    def start_experiment(
        self,
        experiment_name: str,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            description: Experiment description
            parameters: Experiment parameters
            
        Returns:
            Experiment ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        experiment_path = self.experiments_dir / experiment_id
        experiment_path.mkdir(exist_ok=True)
        
        experiment_info = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "description": description,
            "parameters": parameters or {},
            "start_time": timestamp,
            "status": "running",
            "results": {}
        }
        
        info_path = experiment_path / "experiment_info.json"
        with open(info_path, 'w') as f:
            json.dump(experiment_info, f, indent=2, default=str)
        
        logger.info(f"Experiment started: {experiment_id}")
        return experiment_id
    
    def log_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """Log a metric for an experiment.
        
        Args:
            experiment_id: Experiment ID
            metric_name: Name of the metric
            value: Metric value
            step: Optional step number
        """
        experiment_path = self.experiments_dir / experiment_id
        info_path = experiment_path / "experiment_info.json"
        
        try:
            with open(info_path, 'r') as f:
                experiment_info = json.load(f)
            
            if "metrics" not in experiment_info:
                experiment_info["metrics"] = {}
            
            if metric_name not in experiment_info["metrics"]:
                experiment_info["metrics"][metric_name] = []
            
            metric_entry = {
                "value": value,
                "step": step,
                "timestamp": datetime.now().isoformat()
            }
            
            experiment_info["metrics"][metric_name].append(metric_entry)
            
            with open(info_path, 'w') as f:
                json.dump(experiment_info, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error logging metric: {e}")
    
    def save_result(
        self,
        experiment_id: str,
        result_name: str,
        result_data: Any,
        format: str = "pickle"
    ) -> None:
        """Save experiment result.
        
        Args:
            experiment_id: Experiment ID
            result_name: Name of the result
            result_data: Result data
            format: Save format
        """
        experiment_path = self.experiments_dir / experiment_id
        results_dir = experiment_path / "results"
        results_dir.mkdir(exist_ok=True)
        
        try:
            if format == "pickle":
                result_path = results_dir / f"{result_name}.pkl"
                with open(result_path, 'wb') as f:
                    pickle.dump(result_data, f)
            elif format == "json":
                result_path = results_dir / f"{result_name}.json"
                with open(result_path, 'w') as f:
                    json.dump(result_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Result saved: {result_name}")
            
        except Exception as e:
            logger.error(f"Error saving result: {e}")
    
    def finish_experiment(
        self,
        experiment_id: str,
        final_results: Optional[Dict[str, Any]] = None
    ) -> None:
        """Finish an experiment.
        
        Args:
            experiment_id: Experiment ID
            final_results: Final experiment results
        """
        experiment_path = self.experiments_dir / experiment_id
        info_path = experiment_path / "experiment_info.json"
        
        try:
            with open(info_path, 'r') as f:
                experiment_info = json.load(f)
            
            experiment_info["status"] = "completed"
            experiment_info["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if final_results:
                experiment_info["final_results"] = final_results
            
            with open(info_path, 'w') as f:
                json.dump(experiment_info, f, indent=2, default=str)
            
            logger.info(f"Experiment completed: {experiment_id}")
            
        except Exception as e:
            logger.error(f"Error finishing experiment: {e}")
    
    def list_experiments(self) -> list[Dict[str, Any]]:
        """List all experiments.
        
        Returns:
            List of experiment information
        """
        experiments = []
        
        for experiment_dir in self.experiments_dir.iterdir():
            if not experiment_dir.is_dir():
                continue
            
            info_path = experiment_dir / "experiment_info.json"
            if not info_path.exists():
                continue
            
            try:
                with open(info_path, 'r') as f:
                    experiment_info = json.load(f)
                
                experiments.append({
                    "experiment_id": experiment_info.get("experiment_id"),
                    "experiment_name": experiment_info.get("experiment_name"),
                    "status": experiment_info.get("status"),
                    "start_time": experiment_info.get("start_time"),
                    "end_time": experiment_info.get("end_time")
                })
            except Exception as e:
                logger.warning(f"Error reading experiment info {experiment_dir}: {e}")
        
        return sorted(experiments, key=lambda x: x["start_time"], reverse=True)
