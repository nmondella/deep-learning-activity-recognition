"""
Model architectures for Human Activity Recognition.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Any, Tuple
import logging


class ConvLSTMModel:
    """ConvLSTM model for video action recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ConvLSTM model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config['models']['convlstm']
        self.num_classes = len(config['dataset']['selected_classes'])
        self.logger = logging.getLogger(__name__)
    
    def build_model(self) -> keras.Model:
        """
        Build ConvLSTM model.
        
        Returns:
            Compiled Keras model
        """
        input_shape = tuple(self.model_config['input_shape'])
        
        # Input layer
        inputs = keras.Input(shape=input_shape, name='video_input')
        
        # ConvLSTM layers
        x = layers.ConvLSTM2D(
            filters=self.model_config['conv_filters'][0],
            kernel_size=tuple(self.model_config['conv_kernel_size']),
            padding='same',
            return_sequences=True,
            dropout=self.model_config['dropout_rate'],
            recurrent_dropout=self.model_config['recurrent_dropout'],
            name='convlstm_1'
        )(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        
        x = layers.ConvLSTM2D(
            filters=self.model_config['conv_filters'][1],
            kernel_size=tuple(self.model_config['conv_kernel_size']),
            padding='same',
            return_sequences=True,
            dropout=self.model_config['dropout_rate'],
            recurrent_dropout=self.model_config['recurrent_dropout'],
            name='convlstm_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)
        
        x = layers.ConvLSTM2D(
            filters=self.model_config['conv_filters'][2],
            kernel_size=tuple(self.model_config['conv_kernel_size']),
            padding='same',
            return_sequences=True,
            dropout=self.model_config['dropout_rate'],
            recurrent_dropout=self.model_config['recurrent_dropout'],
            name='convlstm_3'
        )(x)
        x = layers.BatchNormalization(name='bn_3')(x)
        
        x = layers.ConvLSTM2D(
            filters=self.model_config['conv_filters'][3],
            kernel_size=tuple(self.model_config['conv_kernel_size']),
            padding='same',
            return_sequences=False,
            dropout=self.model_config['dropout_rate'],
            recurrent_dropout=self.model_config['recurrent_dropout'],
            name='convlstm_4'
        )(x)
        x = layers.BatchNormalization(name='bn_4')(x)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        
        # Dense layers
        x = layers.Dense(
            self.model_config['lstm_units'][0],
            activation='relu',
            name='dense_1'
        )(x)
        x = layers.Dropout(self.model_config['dropout_rate'], name='dropout_1')(x)
        
        x = layers.Dense(
            self.model_config['lstm_units'][1],
            activation='relu',
            name='dense_2'
        )(x)
        x = layers.Dropout(self.model_config['dropout_rate'], name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='ConvLSTM')
        
        self.logger.info(f"Built ConvLSTM model with {self.num_classes} classes")
        
        return model


class LRCNModel:
    """LRCN (Long-term Recurrent Convolutional Networks) model for video action recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LRCN model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_config = config['models']['lrcn']
        self.dataset_config = config['dataset']
        self.num_classes = len(config['dataset']['selected_classes'])
        self.logger = logging.getLogger(__name__)
    
    def build_cnn_feature_extractor(self) -> keras.Model:
        """
        Build CNN feature extractor.
        
        Returns:
            Feature extractor model
        """
        cnn_model_name = self.model_config['cnn_model']
        
        # Load pre-trained CNN
        if cnn_model_name == 'MobileNetV2':
            base_model = keras.applications.MobileNetV2(
                input_shape=(
                    self.dataset_config['image_height'],
                    self.dataset_config['image_width'],
                    self.dataset_config['channels']
                ),
                include_top=False,
                weights='imagenet'
            )
        elif cnn_model_name == 'ResNet50':
            base_model = keras.applications.ResNet50(
                input_shape=(
                    self.dataset_config['image_height'],
                    self.dataset_config['image_width'],
                    self.dataset_config['channels']
                ),
                include_top=False,
                weights='imagenet'
            )
        elif cnn_model_name == 'EfficientNetB0':
            base_model = keras.applications.EfficientNetB0(
                input_shape=(
                    self.dataset_config['image_height'],
                    self.dataset_config['image_width'],
                    self.dataset_config['channels']
                ),
                include_top=False,
                weights='imagenet'
            )
        else:
            raise ValueError(f"Unsupported CNN model: {cnn_model_name}")
        
        # Freeze or unfreeze base model
        base_model.trainable = self.model_config['feature_extractor_trainable']
        
        # Add global average pooling
        feature_extractor = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2)
        ], name='feature_extractor')
        
        return feature_extractor
    
    def build_model(self) -> keras.Model:
        """
        Build LRCN model.
        
        Returns:
            Compiled Keras model
        """
        sequence_length = self.dataset_config['sequence_length']
        
        # Build feature extractor
        feature_extractor = self.build_cnn_feature_extractor()
        feature_dim = feature_extractor.output_shape[-1]
        
        # Input layer
        inputs = keras.Input(
            shape=(
                sequence_length,
                self.dataset_config['image_height'],
                self.dataset_config['image_width'],
                self.dataset_config['channels']
            ),
            name='video_input'
        )
        
        # Apply feature extractor to each frame
        def extract_features(x):
            batch_size = tf.shape(x)[0]
            # Reshape to (batch_size * sequence_length, height, width, channels)
            x = tf.reshape(x, (-1, x.shape[2], x.shape[3], x.shape[4]))
            # Extract features
            features = feature_extractor(x)
            # Reshape back to (batch_size, sequence_length, feature_dim)
            features = tf.reshape(features, (batch_size, sequence_length, feature_dim))
            return features
        
        x = layers.Lambda(extract_features, name='feature_extraction')(inputs)
        
        # LSTM layers
        x = layers.LSTM(
            self.model_config['lstm_units'][0],
            return_sequences=True,
            dropout=self.model_config['dropout_rate'],
            recurrent_dropout=self.model_config['recurrent_dropout'],
            name='lstm_1'
        )(x)
        
        x = layers.LSTM(
            self.model_config['lstm_units'][1],
            return_sequences=False,
            dropout=self.model_config['dropout_rate'],
            recurrent_dropout=self.model_config['recurrent_dropout'],
            name='lstm_2'
        )(x)
        
        # Dense layers
        x = layers.Dense(
            128,
            activation='relu',
            name='dense_1'
        )(x)
        x = layers.Dropout(self.model_config['dropout_rate'], name='dropout_1')(x)
        
        x = layers.Dense(
            64,
            activation='relu',
            name='dense_2'
        )(x)
        x = layers.Dropout(self.model_config['dropout_rate'], name='dropout_2')(x)
        
        # Output layer
        outputs = layers.Dense(
            self.num_classes,
            activation='softmax',
            name='predictions'
        )(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='LRCN')
        
        self.logger.info(f"Built LRCN model with {self.num_classes} classes")
        
        return model


class Attention3DCNN:
    """3D CNN with attention mechanism for video action recognition."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Attention3DCNN model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.dataset_config = config['dataset']
        self.num_classes = len(config['dataset']['selected_classes'])
        self.logger = logging.getLogger(__name__)
    
    def attention_block(self, x: tf.Tensor, filters: int) -> tf.Tensor:
        """
        Attention mechanism block.
        
        Args:
            x: Input tensor
            filters: Number of filters
            
        Returns:
            Attention-weighted tensor
        """
        # Global average pooling to get spatial attention
        gap = layers.GlobalAveragePooling3D()(x)
        gap = layers.Reshape((1, 1, 1, filters))(gap)
        
        # Attention weights
        attention = layers.Dense(filters // 8, activation='relu')(gap)
        attention = layers.Dense(filters, activation='sigmoid')(attention)
        
        # Apply attention
        x_attended = layers.Multiply()([x, attention])
        
        return x_attended
    
    def build_model(self) -> keras.Model:
        """
        Build 3D CNN with attention model.
        
        Returns:
            Compiled Keras model
        """
        input_shape = (
            self.dataset_config['sequence_length'],
            self.dataset_config['image_height'],
            self.dataset_config['image_width'],
            self.dataset_config['channels']
        )
        
        # Input layer
        inputs = keras.Input(shape=input_shape, name='video_input')
        
        # 3D Convolutional layers
        x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((1, 2, 2))(x)
        
        x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = self.attention_block(x, 64)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        
        x = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = self.attention_block(x, 128)
        x = layers.MaxPooling3D((2, 2, 2))(x)
        
        x = layers.Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = self.attention_block(x, 256)
        
        # Global Average Pooling
        x = layers.GlobalAveragePooling3D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs, name='Attention3DCNN')
        
        self.logger.info(f"Built Attention3DCNN model with {self.num_classes} classes")
        
        return model


def create_model(model_type: str, config: Dict[str, Any]) -> keras.Model:
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('convlstm', 'lrcn', 'attention3dcnn')
        config: Configuration dictionary
        
    Returns:
        Compiled Keras model
    """
    if model_type == 'convlstm':
        model_builder = ConvLSTMModel(config)
        return model_builder.build_model()
    elif model_type == 'lrcn':
        model_builder = LRCNModel(config)
        return model_builder.build_model()
    elif model_type == 'attention3dcnn':
        model_builder = Attention3DCNN(config)
        return model_builder.build_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compile_model(model: keras.Model, config: Dict[str, Any]) -> keras.Model:
    """
    Compile the model with specified configuration.
    
    Args:
        model: Keras model to compile
        config: Configuration dictionary
        
    Returns:
        Compiled model
    """
    training_config = config['training']
    
    # Setup optimizer
    if training_config['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=training_config['learning_rate'])
    elif training_config['optimizer'] == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate=training_config['learning_rate'],
            momentum=0.9
        )
    elif training_config['optimizer'] == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=training_config['learning_rate'])
    else:
        raise ValueError(f"Unknown optimizer: {training_config['optimizer']}")
    
    # Setup metrics
    metrics = []
    for metric in training_config['metrics']:
        if metric == 'accuracy':
            metrics.append('accuracy')
        elif metric == 'top_5_accuracy':
            metrics.append(keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy'))
        else:
            metrics.append(metric)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=training_config['loss'],
        metrics=metrics
    )
    
    return model