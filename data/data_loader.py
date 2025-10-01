"""
Data loading and preprocessing utilities for Human Activity Recognition.
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from tqdm import tqdm
import pickle


class VideoDataProcessor:
    """Class for processing video data for human activity recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the VideoDataProcessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sequence_length = config['dataset']['sequence_length']
        self.image_height = config['dataset']['image_height']
        self.image_width = config['dataset']['image_width']
        self.channels = config['dataset']['channels']
        self.classes = config['dataset']['selected_classes']
        self.num_classes = len(self.classes)
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)
        
        self.logger = logging.getLogger(__name__)
    
    def extract_frames(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frame arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            self.logger.warning(f"Could not open video: {video_path}")
            return frames
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            frame_indices = np.linspace(0, frame_count - 1, max_frames, dtype=int)
        else:
            frame_indices = range(frame_count)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame
                frame = cv2.resize(frame, (self.image_width, self.image_height))
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize pixel values
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        
        cap.release()
        return frames
    
    def create_sequence(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Create a sequence of frames for model input.
        
        Args:
            frames: List of frame arrays
            
        Returns:
            Sequence array of shape (sequence_length, height, width, channels)
        """
        if len(frames) < self.sequence_length:
            return None
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
        sequence = np.array([frames[i] for i in frame_indices])
        
        return sequence
    
    def load_dataset(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load and preprocess the video dataset.
        
        Args:
            data_dir: Directory containing video files organized by class
            
        Returns:
            Tuple of (sequences, labels, video_paths)
        """
        sequences = []
        labels = []
        video_paths = []
        
        self.logger.info(f"Loading dataset from {data_dir}")
        
        for class_name in tqdm(self.classes, desc="Processing classes"):
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                self.logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            video_files = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            
            for video_file in tqdm(video_files, desc=f"Processing {class_name}", leave=False):
                video_path = os.path.join(class_dir, video_file)
                
                # Extract frames
                frames = self.extract_frames(video_path, max_frames=50)
                
                if len(frames) >= self.sequence_length:
                    # Create sequence
                    sequence = self.create_sequence(frames)
                    
                    if sequence is not None:
                        sequences.append(sequence)
                        labels.append(class_name)
                        video_paths.append(video_path)
        
        if not sequences:
            raise ValueError("No valid sequences found in the dataset")
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = self.label_encoder.transform(labels)
        
        self.logger.info(f"Loaded {len(X)} sequences from {len(set(labels))} classes")
        self.logger.info(f"Data shape: {X.shape}")
        
        return X, y, video_paths
    
    def create_data_splits(self, X: np.ndarray, y: np.ndarray, 
                          video_paths: List[str]) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input sequences
            y: Labels
            video_paths: Video file paths
            
        Returns:
            Tuple of split data (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train_split = self.config['dataset']['train_split']
        val_split = self.config['dataset']['val_split']
        test_split = self.config['dataset']['test_split']
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation
        val_size = val_split / (train_split + val_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
        )
        
        self.logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to a sequence.
        
        Args:
            sequence: Input sequence
            
        Returns:
            Augmented sequence
        """
        if not self.config['augmentation']['enabled']:
            return sequence
        
        augmented_sequence = sequence.copy()
        
        # Random horizontal flip
        if self.config['augmentation']['horizontal_flip'] and np.random.random() > 0.5:
            augmented_sequence = np.flip(augmented_sequence, axis=2)
        
        # Random brightness adjustment
        brightness_range = self.config['augmentation']['brightness_range']
        brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
        augmented_sequence = np.clip(augmented_sequence * brightness_factor, 0, 1)
        
        return augmented_sequence
    
    def create_data_generator(self, X: np.ndarray, y: np.ndarray, 
                             batch_size: int, shuffle: bool = True, 
                             augment: bool = False) -> tf.data.Dataset:
        """
        Create a TensorFlow data generator.
        
        Args:
            X: Input sequences
            y: Labels
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply augmentation
            
        Returns:
            TensorFlow Dataset
        """
        # Convert labels to categorical
        y_categorical = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((X, y_categorical))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        if augment:
            dataset = dataset.map(
                lambda x, y: (tf.py_function(self.augment_sequence, [x], tf.float32), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def save_processed_data(self, data: Dict[str, Any], save_path: str) -> None:
        """
        Save processed data to file.
        
        Args:
            data: Dictionary containing processed data
            save_path: Path to save the data
        """
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Processed data saved to {save_path}")
    
    def load_processed_data(self, load_path: str) -> Dict[str, Any]:
        """
        Load processed data from file.
        
        Args:
            load_path: Path to load the data from
            
        Returns:
            Dictionary containing processed data
        """
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.logger.info(f"Processed data loaded from {load_path}")
        return data
    
    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        return self.classes
    
    def get_num_classes(self) -> int:
        """Get the number of classes."""
        return self.num_classes
    
    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """Encode string labels to integers."""
        return self.label_encoder.transform(labels)
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """Decode integer labels to strings."""
        return self.label_encoder.inverse_transform(encoded_labels)