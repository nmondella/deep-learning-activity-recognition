"""
Training script for Human Activity Recognition models.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helpers import (
    load_config, setup_logging, create_directories, 
    save_model_info, plot_training_history, set_random_seeds,
    get_gpu_info, save_experiment_results
)
from data.data_loader import VideoDataProcessor
from src.architectures import create_model, compile_model


class ModelTrainer:
    """Class for training Human Activity Recognition models."""
    
    def __init__(self, config: Dict[str, Any], model_type: str):
        """
        Initialize the ModelTrainer.
        
        Args:
            config: Configuration dictionary
            model_type: Type of model to train
        """
        self.config = config
        self.model_type = model_type
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        set_random_seeds(42)
        
        # Setup paths
        self.models_dir = config['paths']['models_dir']
        self.results_dir = config['paths']['results_dir']
        self.logs_dir = config['paths']['logs_dir']
        
        # Create directories
        create_directories([self.models_dir, self.results_dir, self.logs_dir])
        
        # Initialize data processor
        self.data_processor = VideoDataProcessor(config)
        
        # Log GPU info
        gpu_info = get_gpu_info()
        self.logger.info(f"GPU Info: {gpu_info}")
    
    def create_callbacks(self, model_name: str) -> list:
        """
        Create training callbacks.
        
        Args:
            model_name: Name of the model for saving
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_path = os.path.join(self.models_dir, f"{model_name}_best.h5")
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=self.config['training']['model_checkpoint']['monitor'],
            mode=self.config['training']['model_checkpoint']['mode'],
            save_best_only=self.config['training']['model_checkpoint']['save_best_only'],
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor=self.config['training']['early_stopping']['monitor'],
            patience=self.config['training']['early_stopping']['patience'],
            restore_best_weights=self.config['training']['early_stopping']['restore_best_weights'],
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        
        # Reduce learning rate
        reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor=self.config['training']['reduce_lr']['monitor'],
            factor=self.config['training']['reduce_lr']['factor'],
            patience=self.config['training']['reduce_lr']['patience'],
            min_lr=self.config['training']['reduce_lr']['min_lr'],
            verbose=1
        )
        callbacks.append(reduce_lr_callback)
        
        # TensorBoard
        log_dir = os.path.join(self.logs_dir, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False
        )
        callbacks.append(tensorboard_callback)
        
        # Custom callback for saving training metrics
        class MetricsLogger(keras.callbacks.Callback):
            def __init__(self, save_path):
                super().__init__()
                self.save_path = save_path
                self.epoch_metrics = []
            
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                epoch_data = {'epoch': epoch + 1, **logs}
                self.epoch_metrics.append(epoch_data)
                
                # Save metrics after each epoch
                import json
                with open(self.save_path, 'w') as f:
                    json.dump(self.epoch_metrics, f, indent=2)
        
        metrics_path = os.path.join(self.results_dir, f"{model_name}_training_metrics.json")
        metrics_callback = MetricsLogger(metrics_path)
        callbacks.append(metrics_callback)
        
        return callbacks
    
    def load_data(self) -> tuple:
        """
        Load and preprocess the dataset.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_data)
        """
        # Check if processed data exists
        processed_data_path = os.path.join(
            self.config['paths']['processed_data_dir'],
            'processed_data.pkl'
        )
        
        if os.path.exists(processed_data_path):
            self.logger.info("Loading processed data...")
            data = self.data_processor.load_processed_data(processed_data_path)
            X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
            y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
        else:
            self.logger.info("Processing raw data...")
            # Load raw data
            video_dir = os.path.join(self.config['paths']['processed_data_dir'], 'videos')
            
            if not os.path.exists(video_dir):
                raise FileNotFoundError(
                    f"Video directory not found: {video_dir}. "
                    "Please run the data download script first."
                )
            
            # Load and split data
            X, y, video_paths = self.data_processor.load_dataset(video_dir)
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.create_data_splits(
                X, y, video_paths
            )
            
            # Save processed data
            processed_data = {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                'class_names': self.data_processor.get_class_names()
            }
            self.data_processor.save_processed_data(processed_data, processed_data_path)
        
        # Create data generators
        batch_size = self.config['training']['batch_size']
        
        train_dataset = self.data_processor.create_data_generator(
            X_train, y_train, batch_size, shuffle=True, augment=True
        )
        
        val_dataset = self.data_processor.create_data_generator(
            X_val, y_val, batch_size, shuffle=False, augment=False
        )
        
        self.logger.info(f"Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return train_dataset, val_dataset, (X_test, y_test)
    
    def train_model(self) -> tuple:
        """
        Train the model.
        
        Returns:
            Tuple of (model, history, test_data)
        """
        # Load data
        train_dataset, val_dataset, test_data = self.load_data()
        
        # Create model
        self.logger.info(f"Creating {self.model_type} model...")
        model = create_model(self.model_type, self.config)
        model = compile_model(model, self.config)
        
        # Print model summary
        model.summary()
        
        # Save model architecture
        model_name = f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_info_path = os.path.join(self.results_dir, f"{model_name}_architecture.txt")
        save_model_info(model, model_info_path)
        
        # Create callbacks
        callbacks = self.create_callbacks(model_name)
        
        # Train model
        self.logger.info("Starting model training...")
        history = model.fit(
            train_dataset,
            epochs=self.config['training']['epochs'],
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.models_dir, f"{model_name}_final.h5")
        model.save(final_model_path)
        self.logger.info(f"Final model saved to: {final_model_path}")
        
        # Plot training history
        if self.config['visualization']['plot_training_history']:
            history_plot_path = os.path.join(self.results_dir, f"{model_name}_training_history.png")
            plot_training_history(history, history_plot_path)
        
        # Save training results
        training_results = {
            'model_name': model_name,
            'model_type': self.model_type,
            'config': self.config,
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'epochs_trained': len(history.history['accuracy']),
            'model_path': final_model_path
        }
        
        results_path = os.path.join(self.results_dir, f"{model_name}_results.json")
        save_experiment_results(training_results, results_path)
        
        return model, history, test_data
    
    def run_training(self) -> None:
        """Run the complete training pipeline."""
        try:
            self.logger.info(f"Starting training pipeline for {self.model_type} model")
            
            # Train model
            model, history, test_data = self.train_model()
            
            self.logger.info("Training completed successfully!")
            
            # Print final results
            final_val_acc = max(history.history['val_accuracy'])
            final_val_loss = min(history.history['val_loss'])
            
            print(f"\n{'='*50}")
            print(f"Training Results for {self.model_type.upper()}")
            print(f"{'='*50}")
            print(f"Best Validation Accuracy: {final_val_acc:.4f}")
            print(f"Best Validation Loss: {final_val_loss:.4f}")
            print(f"Total Epochs: {len(history.history['accuracy'])}")
            print(f"{'='*50}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise


def main():
    """Main function for training."""
    parser = argparse.ArgumentParser(description='Train Human Activity Recognition model')
    parser.add_argument(
        '--model', 
        choices=['convlstm', 'lrcn', 'attention3dcnn'],
        default='convlstm',
        help='Model type to train'
    )
    parser.add_argument(
        '--config', 
        default='configs/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs to train (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Setup logging
    log_file = os.path.join(
        config['paths']['results_dir'],
        f"training_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    setup_logging(
        config.get('logging', {}).get('level', 'INFO'),
        log_file
    )
    
    # Create trainer and run training
    trainer = ModelTrainer(config, args.model)
    trainer.run_training()


if __name__ == "__main__":
    main()