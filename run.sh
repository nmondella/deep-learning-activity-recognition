#!/bin/bash

# Project management script for Human Activity Recognition

show_help() {
    echo "🎬 Human Activity Recognition - Simple & Clean"
    echo "=============================================="
    echo "Available commands:"
    echo "  ./run.sh train       - Train ConvLSTM model"
    echo "  ./run.sh train-lrcn  - Train LRCN model"
    echo "  ./run.sh train-att   - Train Attention 3D CNN"
    echo "  ./run.sh evaluate    - Evaluate model"  
    echo "  ./run.sh predict     - Run prediction"
    echo "  ./run.sh paper       - Generate research paper PDF"
    echo "  ./run.sh test        - Test setup"
    echo "  ./run.sh clean       - Clean files"
    echo "  ./run.sh info        - Project info"
    echo "  ./run.sh help        - Show this help"
}

case "$1" in
    "train")
        echo "🏋️ Training ConvLSTM model..."
        python3 -m src.train
        ;;
    "train-lrcn")
        echo "🏋️ Training LRCN model..."
        python3 -m src.train --model lrcn
        ;;
    "train-att")
        echo "🏋️ Training Attention 3D CNN model..."
        python3 -m src.train --model attention3dcnn
        ;;
    "evaluate")
        echo "📊 Evaluating model..."
        python3 -m src.evaluate --model_path models/convlstm_best.h5
        ;;
    "predict")
        echo "🔮 Running prediction..."
        python3 -m src.predict --model_path models/convlstm_best.h5
        ;;
    "paper")
        echo "📄 Generating research paper PDF..."
        cd results && python3 compile_paper.py
        ;;
    "test")
        echo "🧪 Testing setup..."
        python3 -c "
import sys
print('✅ Python:', sys.version.split()[0])
try:
    import tensorflow as tf
    print('✅ TensorFlow:', tf.__version__)
except ImportError:
    print('⚠️  TensorFlow not installed - run: pip install tensorflow')
try:
    from src.helpers import setup_logging
    print('✅ Project modules working')
except ImportError as e:
    print('⚠️  Project modules:', str(e))
print('✅ Setup test complete')
"
        ;;
    "clean")
        echo "🧹 Cleaning..."
        find . -name "*.pyc" -delete 2>/dev/null || true
        find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.log" -delete 2>/dev/null || true
        echo "✅ Cleanup complete"
        ;;
    "info")
        echo "🎬 Human Activity Recognition"
        echo "=============================="
        echo "📋 Simple & Clean ML Project"
        echo "👤 Author: Nicholas Mondella"
        echo "🤖 Models: ConvLSTM, LRCN, Attention3D"
        echo "📊 Dataset: UCF101 (25 classes)"
        echo "🔧 Framework: TensorFlow/Keras"
        echo ""
        echo "� Project files:"
        echo "  Python files: $(find src/ -name "*.py" | wc -l | tr -d ' ')"
        echo "  Total files: $(find . -type f -not -path './.git/*' | wc -l | tr -d ' ')"
        ;;
    "help"|*)
        show_help
        ;;
esac