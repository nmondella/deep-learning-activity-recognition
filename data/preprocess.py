"""
Data preprocessing script for Human Activity Recognition.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import load_config, setup_logging
from data.data_loader import VideoDataProcessor


def main():
    """Main function for data preprocessing."""
    parser = argparse.ArgumentParser(description='Preprocess Human Activity Recognition dataset')
    parser.add_argument(
        '--config',
        default='configs/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--input_dir',
        help='Input directory containing video files (overrides config)'
    )
    parser.add_argument(
        '--output_dir',
        help='Output directory for processed data (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.input_dir:
        config['paths']['processed_data_dir'] = args.input_dir
    if args.output_dir:
        config['paths']['processed_data_dir'] = args.output_dir
    
    # Setup logging
    log_file = os.path.join(
        config['paths']['results_dir'],
        f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    setup_logging(
        config.get('logging', {}).get('level', 'INFO'),
        log_file
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing...")
    
    # Initialize data processor
    data_processor = VideoDataProcessor(config)
    
    # Define paths
    video_dir = os.path.join(config['paths']['processed_data_dir'], 'videos')
    processed_data_path = os.path.join(
        config['paths']['processed_data_dir'],
        'processed_data.pkl'
    )
    
    # Check if video directory exists
    if not os.path.exists(video_dir):
        logger.error(f"Video directory not found: {video_dir}")
        logger.error("Please run the dataset download script first.")
        sys.exit(1)
    
    # Check if processed data already exists
    if os.path.exists(processed_data_path):
        logger.warning(f"Processed data already exists: {processed_data_path}")
        response = input("Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            logger.info("Preprocessing cancelled.")
            sys.exit(0)
    
    try:
        # Load and process dataset
        logger.info("Loading raw video data...")
        X, y, video_paths = data_processor.load_dataset(video_dir)
        
        logger.info("Creating data splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.create_data_splits(
            X, y, video_paths
        )
        
        # Prepare processed data
        processed_data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'class_names': data_processor.get_class_names(),
            'config': config
        }
        
        # Save processed data
        logger.info("Saving processed data...")
        data_processor.save_processed_data(processed_data, processed_data_path)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"DATA PREPROCESSING COMPLETED")
        print(f"{'='*60}")
        print(f"Total samples: {len(X)}")
        print(f"Classes: {len(data_processor.get_class_names())}")
        print(f"Train samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Data shape: {X_train.shape}")
        print(f"Processed data saved to: {processed_data_path}")
        print(f"{'='*60}")
        
        logger.info("Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()