#!/usr/bin/env python3
"""
Evaluation utilities for classification/regression problems in alpha factor modeling.
"""

import logging
import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, log_loss
from scipy.stats import pearsonr
from scipy.special import softmax

logger = logging.getLogger(__name__)

def convert_returns_to_classes(returns: np.ndarray, 
                             lower_threshold: float = -0.01, 
                             upper_threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """Convert continuous returns to 3-class classification labels.
    
    Args:
        returns: Array of continuous returns
        lower_threshold: Threshold for negative class (< lower_threshold)
        upper_threshold: Threshold for positive class (> upper_threshold)
        
    Returns:
        Tuple of (class_labels, bin_centers)
        - class_labels: Integer labels (0=negative, 1=neutral, 2=positive)
        - bin_centers: Center values for each class bin
    """
    logger.info(f"Converting {len(returns)} returns to classes with thresholds ({lower_threshold}, {upper_threshold})")
    
    class_labels = np.zeros(len(returns), dtype=int)
    class_labels[returns < lower_threshold] = 0  # Negative
    class_labels[(returns >= lower_threshold) & (returns <= upper_threshold)] = 1  # Neutral
    class_labels[returns > upper_threshold] = 2  # Positive
    
    # Calculate bin centers
    negative_center = np.mean(returns[returns < lower_threshold]) if np.any(returns < lower_threshold) else lower_threshold - 0.005
    neutral_center = np.mean(returns[(returns >= lower_threshold) & (returns <= upper_threshold)]) if np.any((returns >= lower_threshold) & (returns <= upper_threshold)) else 0.0
    positive_center = np.mean(returns[returns > upper_threshold]) if np.any(returns > upper_threshold) else upper_threshold + 0.005
    
    bin_centers = np.array([negative_center, neutral_center, positive_center])
    
    class_counts = np.bincount(class_labels, minlength=3)
    logger.info(f"Class distribution: Negative={class_counts[0]}, Neutral={class_counts[1]}, Positive={class_counts[2]}")
    logger.info(f"Bin centers: Negative={negative_center:.6f}, Neutral={neutral_center:.6f}, Positive={positive_center:.6f}")
    
    return class_labels, bin_centers

def convert_probabilities_to_returns(probabilities: np.ndarray, 
                                   bin_centers: np.ndarray, 
                                   use_softmax: bool = True) -> np.ndarray:
    """Convert classification probabilities back to continuous return predictions.
    
    Args:
        probabilities: Raw model outputs (pre-softmax) or probabilities
        bin_centers: Center values for each class bin
        use_softmax: Whether to apply softmax to probabilities
        
    Returns:
        Array of predicted continuous returns
    """
    logger.debug(f"Converting {len(probabilities)} probability predictions to returns")
    
    if use_softmax:
        # Apply softmax to convert logits to probabilities
        probs = softmax(probabilities, axis=1)
    else:
        probs = probabilities
        
    # Weighted average of bin centers by probability
    predicted_returns = np.dot(probs, bin_centers)
    
    logger.debug(f"Predicted returns range: [{np.min(predicted_returns):.6f}, {np.max(predicted_returns):.6f}]")
    
    return predicted_returns

def evaluate_comprehensive(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         y_pred_probs: Optional[np.ndarray] = None,
                         problem_type: str = "regression",
                         bin_centers: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Comprehensive evaluation for both regression and classification problems.
    
    Args:
        y_true: True values (continuous for regression, class labels for classification)
        y_pred: Predicted values (continuous for regression, class labels for classification)
        y_pred_probs: Predicted probabilities (for classification problems)
        problem_type: "regression" or "classification"
        bin_centers: Bin centers for converting classification to regression proxy
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {problem_type} problem with {len(y_true)} samples")
    
    metrics = {}
    
    if problem_type == "regression":
        # Standard regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        if len(np.unique(y_true)) > 1:  # Avoid division by zero
            metrics['r2'] = r2_score(y_true, y_pred)
            correlation, p_value = pearsonr(y_true.flatten(), y_pred.flatten())
            metrics['correlation'] = correlation
            metrics['correlation_p_value'] = p_value
        else:
            metrics['r2'] = 0.0
            metrics['correlation'] = 0.0
            metrics['correlation_p_value'] = 1.0
            
        metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        metrics['information_ratio'] = metrics['correlation'] / np.std(y_pred) if np.std(y_pred) > 0 else 0.0
        
        logger.info(f"Regression metrics - MSE: {metrics['mse']:.6f}, R²: {metrics['r2']:.6f}, Corr: {metrics['correlation']:.6f}")
        
    elif problem_type == "classification":
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        if y_pred_probs is not None:
            try:
                metrics['log_loss'] = log_loss(y_true, y_pred_probs)
            except ValueError as e:
                logger.warning(f"Could not compute log_loss: {e}")
                metrics['log_loss'] = np.inf
                
            # Convert probabilities to regression proxy and evaluate
            if bin_centers is not None:
                y_pred_continuous = convert_probabilities_to_returns(
                    y_pred_probs, bin_centers, use_softmax=True
                )
                
                # Create continuous targets from class labels
                y_true_continuous = bin_centers[y_true]
                
                # Regression metrics on the proxy problem
                metrics['proxy_mse'] = mean_squared_error(y_true_continuous, y_pred_continuous)
                metrics['proxy_rmse'] = np.sqrt(metrics['proxy_mse'])
                metrics['proxy_mae'] = np.mean(np.abs(y_true_continuous - y_pred_continuous))
                
                if len(np.unique(y_true_continuous)) > 1:
                    metrics['proxy_r2'] = r2_score(y_true_continuous, y_pred_continuous)
                    correlation, p_value = pearsonr(y_true_continuous, y_pred_continuous)
                    metrics['proxy_correlation'] = correlation
                    metrics['proxy_correlation_p_value'] = p_value
                else:
                    metrics['proxy_r2'] = 0.0
                    metrics['proxy_correlation'] = 0.0
                    metrics['proxy_correlation_p_value'] = 1.0
                    
                metrics['proxy_information_ratio'] = metrics['proxy_correlation'] / np.std(y_pred_continuous) if np.std(y_pred_continuous) > 0 else 0.0
                
                logger.info(f"Classification metrics - Accuracy: {metrics['accuracy']:.6f}, Log Loss: {metrics['log_loss']:.6f}")
                logger.info(f"Proxy regression metrics - MSE: {metrics['proxy_mse']:.6f}, R²: {metrics['proxy_r2']:.6f}, Corr: {metrics['proxy_correlation']:.6f}")
        
        # Class distribution analysis
        unique_classes, class_counts = np.unique(y_true, return_counts=True)
        for i, (cls, count) in enumerate(zip(unique_classes, class_counts)):
            metrics[f'class_{cls}_count'] = count
            metrics[f'class_{cls}_proportion'] = count / len(y_true)
            
        logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    
    return metrics

def log_comprehensive_results(results: Dict[str, Dict[str, float]], 
                            problem_type: str = "regression") -> None:
    """Log comprehensive results for all splits (train/val/test).
    
    Args:
        results: Dictionary with keys 'train', 'val', 'test' containing metrics
        problem_type: "regression" or "classification"
    """
    logger.info("="*80)
    logger.info(f"COMPREHENSIVE {problem_type.upper()} RESULTS")
    logger.info("="*80)
    
    for split_name in ['train', 'val', 'test']:
        if split_name not in results:
            continue
            
        metrics = results[split_name]
        logger.info(f"\n{split_name.upper()} SET RESULTS:")
        logger.info("-" * 40)
        
        if problem_type == "regression":
            logger.info(f"MSE: {metrics.get('mse', 'N/A'):.6f}")
            logger.info(f"RMSE: {metrics.get('rmse', 'N/A'):.6f}")
            logger.info(f"MAE: {metrics.get('mae', 'N/A'):.6f}")
            logger.info(f"R²: {metrics.get('r2', 'N/A'):.6f}")
            logger.info(f"Correlation: {metrics.get('correlation', 'N/A'):.6f}")
            logger.info(f"Information Ratio: {metrics.get('information_ratio', 'N/A'):.6f}")
            
        elif problem_type == "classification":
            logger.info(f"Accuracy: {metrics.get('accuracy', 'N/A'):.6f}")
            logger.info(f"Log Loss: {metrics.get('log_loss', 'N/A'):.6f}")
            
            # Class distribution
            for i in range(3):  # Assuming 3-class problem
                count = metrics.get(f'class_{i}_count', 0)
                prop = metrics.get(f'class_{i}_proportion', 0)
                logger.info(f"Class {i}: {count} samples ({prop:.3f})")
            
            # Proxy regression metrics
            if 'proxy_mse' in metrics:
                logger.info(f"\nProxy Regression Metrics:")
                logger.info(f"Proxy MSE: {metrics.get('proxy_mse', 'N/A'):.6f}")
                logger.info(f"Proxy RMSE: {metrics.get('proxy_rmse', 'N/A'):.6f}")
                logger.info(f"Proxy MAE: {metrics.get('proxy_mae', 'N/A'):.6f}")
                logger.info(f"Proxy R²: {metrics.get('proxy_r2', 'N/A'):.6f}")
                logger.info(f"Proxy Correlation: {metrics.get('proxy_correlation', 'N/A'):.6f}")
                logger.info(f"Proxy Info Ratio: {metrics.get('proxy_information_ratio', 'N/A'):.6f}")
    
    # Highlight test set MSE
    if 'test' in results:
        test_mse = results['test'].get('mse') or results['test'].get('proxy_mse')
        if test_mse:
            logger.info("="*80)
            logger.info(f"🎯 FINAL TEST MSE: {test_mse:.6f}")
            logger.info("="*80)