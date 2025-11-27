#!/usr/bin/env python3
"""
Model Inference Engine for loading and running PyTorch models from /data/models/
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class ModelInferenceEngine:
    """Engine for loading and running inference with PyTorch models from /data/models/."""
    
    def __init__(self, models_base_path: str = "/home/lichenhui/data/models"):
        self.models_base_path = Path(models_base_path)
        self.loaded_models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ModelInferenceEngine initialized with device: {self.device}")
    
    def discover_available_models(self) -> Dict[str, str]:
        """Discover all available model Python files.
        
        Returns:
            Dict mapping model names to their file paths
        """
        logger.info(f"Discovering models in {self.models_base_path}")
        
        model_files = {}
        
        # Search for .py files in the models directory structure
        for py_file in self.models_base_path.rglob("*.py"):
            # Skip __init__.py, checkpoints, and test files
            if py_file.name.startswith("__") or "checkpoint" in str(py_file) or py_file.name.startswith("test_"):
                continue
                
            # Extract model name from filename
            model_name = py_file.stem
            model_files[model_name] = str(py_file)
        
        logger.info(f"Discovered {len(model_files)} models: {list(model_files.keys())}")
        return model_files
    
    def load_model_class(self, model_file_path: str, model_name: str) -> type:
        """Dynamically load a model class from a Python file.
        
        Args:
            model_file_path: Path to the Python file containing the model
            model_name: Name of the model class to load
            
        Returns:
            The loaded model class
        """
        logger.info(f"Loading model class {model_name} from {model_file_path}")
        
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(model_name, model_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Try to find the model class (usually matches filename or is capitalized)
            possible_class_names = [
                model_name.upper(),  # LSTM, CNN1, etc.
                model_name.capitalize(),  # Lstm, Cnn1, etc.
                model_name,  # lstm, cnn1, etc.
                'MLP' if model_name == 'mlp' else None,
                'CNN1' if model_name == 'cnn1' else None,
                'CNN2' if model_name == 'cnn2' else None,
                'LSTM' if model_name == 'lstm' else None,
            ]
            
            model_class = None
            for class_name in possible_class_names:
                if class_name and hasattr(module, class_name):
                    model_class = getattr(module, class_name)
                    break
            
            if model_class is None:
                # List all classes in the module
                classes = [name for name in dir(module) 
                          if isinstance(getattr(module, name), type) 
                          and issubclass(getattr(module, name), nn.Module)
                          and name != 'Module']
                
                if classes:
                    model_class = getattr(module, classes[0])
                    logger.info(f"Using class {classes[0]} from module")
                else:
                    raise ValueError(f"No PyTorch model class found in {model_file_path}")
            
            logger.info(f"Successfully loaded model class {model_class.__name__}")
            return model_class
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_file_path}: {e}")
            raise
    
    def create_model_instance(self, model_class: type, model_name: str) -> nn.Module:
        """Create an instance of the model with appropriate parameters.
        
        Args:
            model_class: The loaded model class
            model_name: Name of the model for parameter selection
            
        Returns:
            Instantiated model
        """
        logger.info(f"Creating instance of {model_class.__name__}")
        
        try:
            # Standard parameters for our 25x300 input format
            if model_name.lower() in ['lstm']:
                model = model_class(
                    x_shape=25,  # 25 factors
                    hidden_layer_dim=64,
                    hidden_mlp=128,
                    num_layers=2,
                    p_dropout=0.3
                )
            elif model_name.lower() in ['mlp']:
                model = model_class(
                    num_features=25 * 300,  # Flattened input
                    num_classes=1,  # Single output
                    hidden_layer_dim=128,
                    p_dropout=0.3
                )
            elif model_name.lower() in ['cnn1', 'cnn2']:
                model = model_class(
                    num_features=25,
                    num_classes=1
                )
            else:
                # Generic attempt with common parameters
                try:
                    model = model_class(num_features=25, num_classes=1)
                except TypeError:
                    try:
                        model = model_class(25, 1)  # Positional args
                    except TypeError:
                        model = model_class()  # No args
            
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            logger.info(f"Successfully created {model_class.__name__} instance")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model instance: {e}")
            raise
    
    def load_model(self, model_name: str, model_file_path: Optional[str] = None) -> nn.Module:
        """Load a model for inference.
        
        Args:
            model_name: Name of the model to load
            model_file_path: Optional path to model file (auto-discovered if not provided)
            
        Returns:
            Loaded PyTorch model ready for inference
        """
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded, returning cached version")
            return self.loaded_models[model_name]
        
        if model_file_path is None:
            available_models = self.discover_available_models()
            if model_name not in available_models:
                raise ValueError(f"Model {model_name} not found. Available: {list(available_models.keys())}")
            model_file_path = available_models[model_name]
        
        # Load the model class and create instance
        model_class = self.load_model_class(model_file_path, model_name)
        model = self.create_model_instance(model_class, model_name)
        
        # Cache the model
        self.loaded_models[model_name] = model
        
        logger.info(f"Model {model_name} loaded and cached successfully")
        return model
    
    def predict(self, model: nn.Module, input_data: np.ndarray, 
               output_type: str = "regression") -> np.ndarray:
        """Run inference with a loaded model.
        
        Args:
            model: Loaded PyTorch model
            input_data: Input data array, shape depends on model type
            output_type: "regression" or "classification"
            
        Returns:
            Model predictions as numpy array
        """
        logger.debug(f"Running inference with input shape: {input_data.shape}")
        
        try:
            # Convert to tensor
            if len(input_data.shape) == 2:  # Single sample (300, 25)
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
            else:  # Batch (N, 300, 25)
                input_tensor = torch.FloatTensor(input_data).to(self.device)
            
            model.eval()
            with torch.no_grad():
                # Models expect 4 arguments but only use the first
                outputs = model(input_tensor, None, None, None)
                
                if output_type == "classification":
                    # Return raw logits for classification
                    predictions = outputs.cpu().numpy()
                else:
                    # Return single values for regression
                    predictions = outputs.cpu().numpy().flatten()
            
            logger.debug(f"Generated predictions with shape: {predictions.shape}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
    
    def predict_with_model_name(self, model_name: str, input_data: np.ndarray,
                               output_type: str = "regression") -> np.ndarray:
        """Load model and run prediction in one call.
        
        Args:
            model_name: Name of the model to use
            input_data: Input data for prediction
            output_type: "regression" or "classification"
            
        Returns:
            Model predictions
        """
        model = self.load_model(model_name)
        return self.predict(model, input_data, output_type)
    
    def list_available_models(self) -> List[str]:
        """List all available models."""
        return list(self.discover_available_models().keys())
    
    def evaluate_model(self, model_name: str, test_data: np.ndarray, test_targets: np.ndarray,
                      output_type: str = "regression") -> Dict[str, float]:
        """Evaluate a model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            test_data: Test input data
            test_targets: Test target values
            output_type: "regression" or "classification"
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model {model_name} on {len(test_data)} samples")
        
        # Run predictions
        predictions = self.predict_with_model_name(model_name, test_data, output_type)
        
        # Import evaluation utilities
        try:
            from evaluation_utils import evaluate_comprehensive, convert_returns_to_classes
            
            if output_type == "regression":
                metrics = evaluate_comprehensive(test_targets, predictions, problem_type="regression")
            else:
                # For classification, need to convert probabilities and compute metrics
                class_targets, bin_centers = convert_returns_to_classes(test_targets)
                pred_classes = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else predictions
                metrics = evaluate_comprehensive(
                    class_targets, pred_classes, predictions,
                    problem_type="classification", bin_centers=bin_centers
                )
            
            logger.info(f"Model {model_name} evaluation completed")
            return metrics
            
        except ImportError:
            logger.warning("Could not import evaluation utilities, using basic metrics")
            from sklearn.metrics import mean_squared_error, r2_score
            
            if output_type == "regression":
                mse = mean_squared_error(test_targets, predictions)
                r2 = r2_score(test_targets, predictions)
                return {'mse': mse, 'r2': r2}
            else:
                from sklearn.metrics import accuracy_score
                pred_classes = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else predictions
                accuracy = accuracy_score(test_targets, pred_classes)
                return {'accuracy': accuracy}