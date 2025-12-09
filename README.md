# Underwater Fish Anomaly Detection

**Automated health and behavioral anomaly detection for aquaculture monitoring**

## Overview

This project develops a computer vision pipeline to detect anomalies in fish behavior and appearance from underwater video footage. The system uses unsupervised learning to identify potential health issues, abnormal swimming patterns, and environmental stress indicators—critical capabilities for precision aquaculture and early intervention in fish farming operations.

### Real-World Application

Commercial fish farming operations monitor thousands of fish daily. Manual observation is time-intensive, subjective, and often catches health issues too late. This automated system provides:
- **Early health detection**: Identifies abnormal appearance or behavior before visible symptoms
- **Scalable monitoring**: Processes continuous video streams across multiple pens/tanks
- **Data-driven insights**: Quantifies behavioral patterns for feeding optimization and welfare assessment

### Technical Approach

The pipeline combines classical computer vision with modern deep learning:
1. **Preprocessing**: Underwater image enhancement (color correction, contrast adjustment)
2. **Detection & Tracking**: Multi-object tracking to follow individual fish across frames
3. **Feature Engineering**: Extract behavioral (motion patterns, trajectory) and visual (appearance, posture) features
4. **Anomaly Detection**: Unsupervised methods (Isolation Forest, Autoencoders, One-Class SVM) to identify outliers
5. **Validation**: Statistical analysis of detected anomalies and model performance metrics

## Project Status

🚧 **In Development** - Currently implementing baseline pipeline

- [x] Project setup and documentation
- [ ] Data acquisition and preprocessing
- [ ] Fish detection and tracking module
- [ ] Feature extraction pipeline
- [ ] Anomaly detection models
- [ ] Interactive visualization dashboard
- [ ] Performance benchmarking

## Dataset

Using the [DeepFish](https://alzayats.github.io/DeepFish/) dataset and/or aquarium footage for development. For production-scale validation, targeting underwater fish farming video (if accessible).

**Key challenges addressed:**
- Varying underwater visibility conditions
- Occlusions and overlapping fish
- Scale variation (distance from camera)
- Species-specific normal behavior baselines

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/fish-anomaly-detection.git
cd fish-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run preprocessing on sample video
python src/preprocess.py --input data/raw/sample_video.mp4 --output data/processed/

# Train anomaly detection model
python src/train.py --config configs/baseline.yaml

# Run inference on new footage
python src/detect_anomalies.py --video data/test/farm_footage.mp4 --model models/best_model.pth

# Launch interactive dashboard
streamlit run app.py
```

## Project Structure

```
fish-anomaly-detection/
├── README.md
├── CLAUDE.md              # Development context for AI assistants
├── requirements.txt
├── configs/
│   └── baseline.yaml      # Model and pipeline configurations
├── data/
│   ├── raw/              # Original video footage
│   ├── processed/        # Preprocessed frames/features
│   └── annotations/      # Ground truth labels (if available)
├── src/
│   ├── preprocess.py     # Image enhancement and frame extraction
│   ├── detection.py      # Fish detection and tracking
│   ├── features.py       # Feature engineering
│   ├── models.py         # Anomaly detection models
│   ├── train.py          # Training pipeline
│   └── detect_anomalies.py  # Inference script
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory data analysis
│   ├── 02_baseline.ipynb # Initial model experiments
│   └── 03_evaluation.ipynb  # Performance analysis
├── models/               # Saved model weights
├── outputs/              # Detection results and visualizations
└── app.py               # Streamlit dashboard
```

## Key Features

### 1. Behavioral Anomalies
- Abnormal swimming patterns (erratic movement, lethargy)
- Trajectory deviations from population norms
- Temporal changes in activity levels

### 2. Visual Anomalies
- Abnormal body posture or orientation
- Color/texture changes indicating stress or disease
- Size/shape deviations from expected distribution

### 3. Environmental Context
- Spatial distribution patterns (clustering, isolation)
- Group behavior dynamics
- Response to feeding events

## Technical Stack

- **Computer Vision**: OpenCV, scikit-image
- **Deep Learning**: PyTorch, torchvision
- **Object Detection/Tracking**: YOLO, DeepSORT
- **ML/Statistics**: scikit-learn, scipy, statsmodels
- **Visualization**: matplotlib, seaborn, Streamlit
- **Data Processing**: pandas, NumPy

## Performance Metrics

Model evaluation focuses on:
- **Precision/Recall**: Balancing false alarms vs. missed detections
- **Detection Latency**: Real-time processing requirements
- **Robustness**: Performance across varying water clarity and lighting
- **Interpretability**: Understanding what features trigger anomalies

## Background

This project applies quantitative analysis techniques from neuroscience research to aquaculture monitoring. The core methodology—identifying regime-specific patterns in time-series data—transfers naturally from neural recordings to fish behavioral analysis.

**Author**: Raymond Murray  
**Context**: Portfolio project demonstrating computer vision and anomaly detection for ML Engineer positions  
**Related Experience**: 3+ years analyzing multi-modal time-series data, feature engineering for behavioral state classification

## Future Enhancements

- Multi-camera fusion for 3D trajectory reconstruction
- Transfer learning from laboratory to commercial farm settings
- Integration with environmental sensors (temperature, oxygen, pH)
- Real-time alerting system for farm operators
- Species-specific model fine-tuning

## Contributing

This is a portfolio project, but feedback and suggestions are welcome! Please open an issue to discuss potential improvements.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- DeepFish dataset creators
- Aquaculture domain experts who provided insights on fish health indicators
- Computer vision community for open-source tools and pretrained models

---

**Contact**: raymond.don.murray@gmail.com | [LinkedIn](https://linkedin.com/in/yourprofile) | [Portfolio](https://yourportfolio.com)