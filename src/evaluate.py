"""
Evaluation script for Human Activity Recognition models.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helpers import (
    load_config, setup_logging, plot_confusion_matrix,
    save_classification_report, save_experiment_results
)
from data.data_loader import VideoDataProcessor


class ModelEvaluator:
    """Class for evaluating Human Activity Recognition models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelEvaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize data processor
        self.data_processor = VideoDataProcessor(config)
        
        # Setup paths
        self.results_dir = config['paths']['results_dir']
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load test data.
        
        Returns:
            Tuple of (X_test, y_test)
        """
        # Check if processed data exists
        processed_data_path = os.path.join(
            self.config['paths']['processed_data_dir'],
            'processed_data.pkl'
        )
        
        if os.path.exists(processed_data_path):
            self.logger.info("Loading processed test data...")
            data = self.data_processor.load_processed_data(processed_data_path)
            X_test, y_test = data['X_test'], data['y_test']
        else:
            raise FileNotFoundError(
                f"Processed data not found: {processed_data_path}. "
                "Please run the data preprocessing script first."
            )
        
        self.logger.info(f"Test data loaded: {len(X_test)} samples")
        return X_test, y_test
    
    def evaluate_model(self, model_path: str) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Evaluating model: {model_path}")
        
        # Load model
        try:
            model = keras.models.load_model(model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
        
        # Load test data
        X_test, y_test = self.load_test_data()
        
        # Make predictions
        self.logger.info("Making predictions...")
        y_pred_proba = model.predict(X_test, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_test, y_pred, average=None)
        
        # Top-k accuracy
        top_5_accuracy = self.calculate_top_k_accuracy(y_test, y_pred_proba, k=5)
        
        # Class names
        class_names = self.data_processor.get_class_names()
        
        # Create results dictionary
        results = {
            'model_path': model_path,
            'test_samples': len(X_test),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'top_5_accuracy': float(top_5_accuracy),
            'per_class_metrics': {
                class_names[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                }
                for i in range(len(class_names))
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist()
            }
        }
        
        # Generate model name for saving results
        model_name = os.path.basename(model_path).replace('.h5', '')
        
        # Save confusion matrix plot
        if self.config['evaluation']['save_confusion_matrix']:
            cm_path = os.path.join(self.results_dir, f"{model_name}_confusion_matrix.png")
            plot_confusion_matrix(y_test, y_pred, class_names, cm_path)
        
        # Save classification report
        if self.config['evaluation']['save_classification_report']:
            report_path = os.path.join(self.results_dir, f"{model_name}_classification_report.txt")
            save_classification_report(y_test, y_pred, class_names, report_path)
        
        # Save predictions
        if self.config['evaluation']['save_predictions']:
            predictions_path = os.path.join(self.results_dir, f"{model_name}_predictions.csv")
            self.save_predictions_csv(y_test, y_pred, y_pred_proba, class_names, predictions_path)
        
        # Save evaluation results
        results_path = os.path.join(self.results_dir, f"{model_name}_evaluation_results.json")
        save_experiment_results(results, results_path)
        
        self.logger.info(f"Evaluation completed. Results saved to: {results_path}")
        
        return results
    
    def calculate_top_k_accuracy(self, y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 5) -> float:
        """
        Calculate top-k accuracy.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            k: Value of k for top-k accuracy
            
        Returns:
            Top-k accuracy score
        """
        top_k_predictions = np.argsort(y_pred_proba, axis=1)[:, -k:]
        correct = 0
        
        for i, true_label in enumerate(y_true):
            if true_label in top_k_predictions[i]:
                correct += 1
        
        return correct / len(y_true)
    
    def save_predictions_csv(self, y_true: np.ndarray, y_pred: np.ndarray,
                            y_pred_proba: np.ndarray, class_names: list, save_path: str) -> None:
        """
        Save predictions to CSV file.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            class_names: List of class names
            save_path: Path to save CSV file
        """
        # Create DataFrame
        data = {
            'true_label': [class_names[label] for label in y_true],
            'predicted_label': [class_names[label] for label in y_pred],
            'correct': y_true == y_pred,
            'confidence': np.max(y_pred_proba, axis=1)
        }
        
        # Add probability for each class
        for i, class_name in enumerate(class_names):
            data[f'prob_{class_name}'] = y_pred_proba[:, i]
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        
        self.logger.info(f"Predictions saved to: {save_path}")
    
    def compare_models(self, model_paths: list) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            model_paths: List of model paths to compare
            
        Returns:
            Dictionary containing comparison results
        """
        self.logger.info(f"Comparing {len(model_paths)} models...")
        
        comparison_results = {}
        
        for model_path in model_paths:
            model_name = os.path.basename(model_path).replace('.h5', '')
            
            try:
                results = self.evaluate_model(model_path)
                comparison_results[model_name] = {
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score'],
                    'top_5_accuracy': results['top_5_accuracy']
                }
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_path}: {str(e)}")
                comparison_results[model_name] = None
        
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison_results).T
        df_comparison = df_comparison.round(4)
        
        # Save comparison results
        comparison_path = os.path.join(
            self.results_dir,
            f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        df_comparison.to_csv(comparison_path)
        
        # Save as JSON
        comparison_json_path = comparison_path.replace('.csv', '.json')
        save_experiment_results(comparison_results, comparison_json_path)
        
        self.logger.info(f"Model comparison saved to: {comparison_path}")
        
        return comparison_results
    
    def print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation summary.
        
        Args:
            results: Evaluation results dictionary
        """
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Model: {os.path.basename(results['model_path'])}")
        print(f"Test Samples: {results['test_samples']}")
        print(f"{'='*60}")
        print(f"Overall Metrics:")
        print(f"  Accuracy:      {results['accuracy']:.4f}")
        print(f"  Precision:     {results['precision']:.4f}")
        print(f"  Recall:        {results['recall']:.4f}")
        print(f"  F1-Score:      {results['f1_score']:.4f}")
        print(f"  Top-5 Accuracy: {results['top_5_accuracy']:.4f}")
        print(f"{'='*60}")
        
        # Top performing classes
        per_class = results['per_class_metrics']
        sorted_classes = sorted(per_class.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        print(f"Top 5 Performing Classes (by F1-Score):")
        for i, (class_name, metrics) in enumerate(sorted_classes[:5]):
            print(f"  {i+1}. {class_name}: {metrics['f1_score']:.4f}")
        
        print(f"\nWorst 5 Performing Classes (by F1-Score):")
        for i, (class_name, metrics) in enumerate(sorted_classes[-5:]):
            print(f"  {i+1}. {class_name}: {metrics['f1_score']:.4f}")
        
        print(f"{'='*60}")


def main():
    """Main function for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate Human Activity Recognition model')
    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--config',
        default='configs/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Paths to multiple models for comparison'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_file = os.path.join(
        config['paths']['results_dir'],
        f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    setup_logging(
        config.get('logging', {}).get('level', 'INFO'),
        log_file
    )
    
    # Create evaluator
    evaluator = ModelEvaluator(config)
    
    if args.compare:
        # Compare multiple models
        comparison_results = evaluator.compare_models(args.compare)
        
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON RESULTS")
        print(f"{'='*60}")
        
        df = pd.DataFrame(comparison_results).T
        print(df.round(4))
        
    else:
        # Evaluate single model
        results = evaluator.evaluate_model(args.model_path)
        evaluator.print_evaluation_summary(results)


if __name__ == "__main__":
    main()