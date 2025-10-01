# 🎬 Human Activity Recognition

Clean deep learning for video activity recognition.

## 🚀 Quick Start
```bash
pip install -r requirements.txt
./run.sh train
./run.sh paper
```

## 🎯 Features
- 3 CNN+LSTM models (ConvLSTM, LRCN, Attention3D)
- UCF101 dataset (25 activity classes)
- Complete ML pipeline
- Automatic paper generation

## 📁 Structure
```
├── src/        # Source code (5 files)
├── data/       # Dataset processing (3 files)  
├── results/    # Research paper (3 files)
├── models/     # Trained models
└── run.sh      # Automation script
```

## 🛠️ Commands
```bash
./run.sh train      # Train models
./run.sh evaluate   # Evaluate performance
./run.sh predict    # Run predictions
./run.sh paper      # Generate PDF paper
./run.sh info       # Project stats
```

## 📊 Expected Results
| Model | Accuracy |
|-------|----------|
| Attention3D | ~87.1% |
| ConvLSTM | ~85.2% |
| LRCN | ~82.7% |

---
**Author**: Nicholas Mondella  
**Framework**: TensorFlow/Keras
