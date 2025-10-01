"""
Utility functions for the Human Activity Recognition project.
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to save logs
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    return logging.getLogger(__name__)


def create_directories(paths: List[str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_model_info(model: tf.keras.Model, save_path: str) -> None:
    """
    Save model architecture and summary to file.
    
    Args:
        model: Keras model
        save_path: Path to save model info
    """
    with open(save_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def plot_training_history(history: tf.keras.callbacks.History, 
                         save_path: str = None) -> None:
    """
    Plot training history including loss and accuracy.
    
    Args:
        history: Keras training history
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training & validation loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot training & validation accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], save_path: str = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Optional path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: List[str], save_path: str) -> None:
    """
    Save classification report to file.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the report
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    
    with open(save_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)


def calculate_model_size(model_path: str) -> Tuple[float, str]:
    """
    Calculate model file size.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Tuple of (size, unit)
    """
    size_bytes = os.path.getsize(model_path)
    
    if size_bytes < 1024:
        return size_bytes, "B"
    elif size_bytes < 1024**2:
        return size_bytes / 1024, "KB"
    elif size_bytes < 1024**3:
        return size_bytes / (1024**2), "MB"
    else:
        return size_bytes / (1024**3), "GB"


def count_model_parameters(model: tf.keras.Model) -> Tuple[int, int]:
    """
    Count model parameters.
    
    Args:
        model: Keras model
        
    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable_params = sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    total_params = sum([np.prod(v.get_shape()) for v in model.weights])
    
    return int(trainable_params), int(total_params)


def save_experiment_results(results: Dict[str, Any], save_path: str) -> None:
    """
    Save experiment results to JSON file.
    
    Args:
        results: Dictionary containing experiment results
        save_path: Path to save results
    """
    import json
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4, default=str)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    import random
    random.seed(seed)


def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU information.
    
    Returns:
        Dictionary containing GPU information
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    gpu_info = {
        'num_gpus': len(gpus),
        'gpu_names': [gpu.name for gpu in gpus],
        'gpu_available': len(gpus) > 0
    }
    
    if gpu_info['gpu_available']:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    return gpu_info