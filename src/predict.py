"""
Prediction script for Human Activity Recognition models.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helpers import load_config, setup_logging
from data.data_loader import VideoDataProcessor


class ActionPredictor:
    """Class for making predictions on video files."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        Initialize the ActionPredictor.
        
        Args:
            model_path: Path to trained model
            config: Configuration dictionary
        """
        self.model_path = model_path
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = self.load_model()
        
        # Initialize data processor
        self.data_processor = VideoDataProcessor(config)
        
        # Get class names
        self.class_names = self.data_processor.get_class_names()
        self.num_classes = len(self.class_names)
        
        self.logger.info(f"Predictor initialized with {self.num_classes} classes")
    
    def load_model(self) -> keras.Model:
        """
        Load the trained model.
        
        Returns:
            Loaded Keras model
        """
        try:
            model = keras.models.load_model(self.model_path)
            self.logger.info(f"Model loaded from: {self.model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess_video(self, video_path: str) -> np.ndarray:
        """
        Preprocess video for prediction.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed video sequence
        """
        # Extract frames using data processor
        frames = self.data_processor.extract_frames(video_path, max_frames=50)
        
        if len(frames) < self.data_processor.sequence_length:
            raise ValueError(
                f"Video has only {len(frames)} frames, "
                f"but model requires at least {self.data_processor.sequence_length} frames"
            )
        
        # Create sequence
        sequence = self.data_processor.create_sequence(frames)
        
        if sequence is None:
            raise ValueError("Failed to create sequence from video")
        
        # Add batch dimension
        sequence = np.expand_dims(sequence, axis=0)
        
        return sequence
    
    def predict_video(self, video_path: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Predict action for a video file.
        
        Args:
            video_path: Path to video file
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing prediction results
        """
        self.logger.info(f"Predicting action for: {video_path}")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Preprocess video
        try:
            sequence = self.preprocess_video(video_path)
        except Exception as e:
            self.logger.error(f"Failed to preprocess video: {str(e)}")
            raise
        
        # Make prediction
        try:
            predictions = self.model.predict(sequence, verbose=0)[0]
        except Exception as e:
            self.logger.error(f"Failed to make prediction: {str(e)}")
            raise
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            top_predictions.append({
                'class': self.class_names[idx],
                'confidence': float(predictions[idx]),
                'index': int(idx)
            })
        
        # Prepare results
        results = {
            'video_path': video_path,
            'predicted_class': self.class_names[np.argmax(predictions)],
            'confidence': float(np.max(predictions)),
            'top_predictions': top_predictions,
            'all_predictions': {
                self.class_names[i]: float(predictions[i])
                for i in range(len(self.class_names))
            }
        }
        
        self.logger.info(f"Prediction completed. Top class: {results['predicted_class']} ({results['confidence']:.4f})")
        
        return results
    
    def predict_batch(self, video_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Predict actions for multiple videos.
        
        Args:
            video_paths: List of video file paths
            
        Returns:
            List of prediction results
        """
        results = []
        
        for video_path in video_paths:
            try:
                result = self.predict_video(video_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to predict {video_path}: {str(e)}")
                results.append({
                    'video_path': video_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_webcam(self, duration: int = 10) -> Dict[str, Any]:
        """
        Predict action from webcam feed.
        
        Args:
            duration: Duration in seconds to capture video
            
        Returns:
            Prediction results
        """
        self.logger.info(f"Capturing video from webcam for {duration} seconds...")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        # Set frame rate
        fps = 30
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        frames = []
        frame_count = 0
        target_frames = duration * fps
        
        try:
            while frame_count < target_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Resize and normalize frame
                frame = cv2.resize(
                    frame,
                    (self.data_processor.image_width, self.data_processor.image_height)
                )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
                frame_count += 1
                
                # Show frame (optional)
                display_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.putText(
                    display_frame,
                    f"Recording... {frame_count}/{target_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                cv2.imshow('Webcam Feed', display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if len(frames) < self.data_processor.sequence_length:
            raise ValueError(
                f"Captured only {len(frames)} frames, "
                f"but model requires at least {self.data_processor.sequence_length} frames"
            )
        
        # Create sequence
        sequence = self.data_processor.create_sequence(frames)
        sequence = np.expand_dims(sequence, axis=0)
        
        # Make prediction
        predictions = self.model.predict(sequence, verbose=0)[0]
        
        # Get top-5 predictions
        top_indices = np.argsort(predictions)[-5:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            top_predictions.append({
                'class': self.class_names[idx],
                'confidence': float(predictions[idx]),
                'index': int(idx)
            })
        
        results = {
            'source': 'webcam',
            'duration': duration,
            'frames_captured': len(frames),
            'predicted_class': self.class_names[np.argmax(predictions)],
            'confidence': float(np.max(predictions)),
            'top_predictions': top_predictions
        }
        
        return results


def main():
    """Main function for prediction."""
    parser = argparse.ArgumentParser(description='Predict human activity in videos')
    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--video_path',
        help='Path to video file for prediction'
    )
    parser.add_argument(
        '--video_dir',
        help='Directory containing multiple videos'
    )
    parser.add_argument(
        '--webcam',
        action='store_true',
        help='Use webcam for real-time prediction'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Duration in seconds for webcam capture'
    )
    parser.add_argument(
        '--config',
        default='configs/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top predictions to show'
    )
    parser.add_argument(
        '--output',
        help='Output file to save results (JSON format)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config.get('logging', {}).get('level', 'INFO'))
    
    # Create predictor
    predictor = ActionPredictor(args.model_path, config)
    
    results = None
    
    if args.webcam:
        # Webcam prediction
        results = predictor.predict_from_webcam(args.duration)
        
        print(f"\n{'='*50}")
        print(f"WEBCAM PREDICTION RESULTS")
        print(f"{'='*50}")
        print(f"Predicted Action: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.4f}")
        print(f"Frames Captured: {results['frames_captured']}")
        print(f"\nTop {len(results['top_predictions'])} Predictions:")
        for i, pred in enumerate(results['top_predictions']):
            print(f"  {i+1}. {pred['class']}: {pred['confidence']:.4f}")
        print(f"{'='*50}")
        
    elif args.video_path:
        # Single video prediction
        results = predictor.predict_video(args.video_path, args.top_k)
        
        print(f"\n{'='*50}")
        print(f"VIDEO PREDICTION RESULTS")
        print(f"{'='*50}")
        print(f"Video: {os.path.basename(results['video_path'])}")
        print(f"Predicted Action: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.4f}")
        print(f"\nTop {len(results['top_predictions'])} Predictions:")
        for i, pred in enumerate(results['top_predictions']):
            print(f"  {i+1}. {pred['class']}: {pred['confidence']:.4f}")
        print(f"{'='*50}")
        
    elif args.video_dir:
        # Multiple videos prediction
        video_paths = []
        for file in os.listdir(args.video_dir):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(args.video_dir, file))
        
        if not video_paths:
            print(f"No video files found in {args.video_dir}")
            return
        
        results = predictor.predict_batch(video_paths)
        
        print(f"\n{'='*60}")
        print(f"BATCH PREDICTION RESULTS")
        print(f"{'='*60}")
        print(f"Total Videos: {len(video_paths)}")
        print(f"{'='*60}")
        
        for i, result in enumerate(results):
            if 'error' in result:
                print(f"{i+1}. {os.path.basename(result['video_path'])}: ERROR - {result['error']}")
            else:
                print(f"{i+1}. {os.path.basename(result['video_path'])}: {result['predicted_class']} ({result['confidence']:.4f})")
        
        print(f"{'='*60}")
        
    else:
        print("Please specify --video_path, --video_dir, or --webcam")
        return
    
    # Save results if output file specified
    if args.output and results:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()