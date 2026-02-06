# UAV Thermal Object Detection

Detection &amp; OOD: Developed thermal imagery pipelines with metric learning for out-of-distribution identification.

## Overview

This repository contains implementations for two key assignments:

1. **YOLOv8-based Thermal Object Detection Pipeline**: A complete PyTorch/OpenCV pipeline for detecting objects in thermal imagery with specialized augmentations and comprehensive mAP evaluation.

2. **Triplet Loss Mathematical Analysis**: Analytical derivations proving the effect of margin in Triplet Loss on inter-class distances and why OOD samples reside further from class prototypes in metric embedding spaces.

## Features

### Assignment 1: Thermal Object Detection
- ✅ YOLOv8 integration for state-of-the-art detection
- ✅ Thermal-specific data augmentations (histogram equalization, thermal drift simulation, atmospheric effects)
- ✅ Comprehensive mAP evaluation (mAP@0.5, mAP@0.75, mAP@0.5:0.95)
- ✅ COCO-style metrics and per-class AP
- ✅ Visualization and result export
- ✅ Support for mosaic and mixup augmentations
- ✅ Configurable pipeline via YAML

### Assignment 2: Mathematical Proofs
- ✅ Analytical derivation of margin effects in Triplet Loss
- ✅ Proof of inter-class distance relationships
- ✅ Theoretical analysis of OOD sample positioning
- ✅ Quantitative bounds and geometric interpretations
- ✅ Practical implications for OOD detection

## Installation

```bash
# Clone the repository
git clone https://github.com/deekshithj1021/UAV-Thermal-Object-Detection.git
cd UAV-Thermal-Object-Detection

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Training on Thermal Dataset

```python
from thermal_detector import ThermalObjectDetector

# Initialize detector
detector = ThermalObjectDetector('config.yaml')

# Load pretrained YOLOv8
detector.load_model('yolov8n.pt')

# Train on your thermal dataset
detector.train(
    data_yaml_path='data.yaml',
    epochs=100,
    batch_size=16
)
```

### 2. Object Detection

```python
# Detect objects in thermal image
detections = detector.detect(
    'path/to/thermal_image.jpg',
    conf_threshold=0.25,
    visualize=True
)

print(f"Detected {len(detections['boxes'])} objects")
```

### 3. Model Evaluation

```python
# Evaluate model and calculate mAP
results = detector.evaluate(
    test_data_path='data/thermal/test/images',
    ground_truth_path='data/thermal/test/labels'
)

# Results include mAP@0.5, mAP@0.75, mAP@0.5:0.95
```

## Dataset Format

The pipeline expects YOLO format:

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Label format: `class_id x_center y_center width height` (normalized 0-1)

## Configuration

Edit `config.yaml` to customize:
- Model parameters (size, thresholds)
- Training hyperparameters
- Augmentation settings
- Evaluation metrics
- Output options

## Thermal-Specific Augmentations

Implemented augmentations tailored for thermal imagery:

1. **Histogram Equalization (CLAHE)**: Enhances thermal contrast
2. **Thermal Drift Simulation**: Models sensor temperature drift
3. **Atmospheric Effects**: Gaussian noise, blur, fog
4. **Motion Blur**: Simulates UAV movement
5. **Geometric**: Rotation, translation, scaling, perspective
6. **Mosaic/MixUp**: Advanced augmentations for small objects

## mAP Evaluation

Comprehensive evaluation metrics:
- **mAP@0.5**: PASCAL VOC metric
- **mAP@0.75**: Stricter IoU threshold
- **mAP@0.5:0.95**: COCO-style metric (averaged over 10 IoU thresholds)
- **Per-class AP**: Individual class performance

## Mathematical Analysis

See [`TRIPLET_LOSS_ANALYSIS.md`](TRIPLET_LOSS_ANALYSIS.md) for detailed proofs including:

### Theorem 1: Margin Effect
The margin $m$ in Triplet Loss establishes a minimum inter-class distance:
$$d_{\text{inter}} \geq d_{\text{intra}} + m$$

### Theorem 3: OOD Positioning
OOD samples reside at distance:
$$d_{\text{OOD}} > \max_i \sigma_i + \frac{m}{2}$$

from nearest class prototypes, where $\sigma_i$ is intra-class standard deviation.

## File Structure

```
├── thermal_detector.py         # Main YOLOv8 detection pipeline
├── thermal_augmentation.py     # Thermal-specific augmentations
├── map_evaluation.py           # mAP calculation module
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── example_usage.py            # Usage examples
├── TRIPLET_LOSS_ANALYSIS.md   # Mathematical proofs
└── README.md                   # This file
```

## Examples

Run example scripts:

```bash
python example_usage.py
```

This demonstrates:
1. Training workflow
2. Inference on thermal images
3. mAP evaluation
4. Data augmentation
5. Custom dataset preparation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy, Matplotlib
- Albumentations

See `requirements.txt` for complete list.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{uav-thermal-detection,
  title={UAV Thermal Object Detection with OOD Analysis},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/deekshithj1021/UAV-Thermal-Object-Detection}}
}
```

## License

This project is available for educational and research purposes.

## Acknowledgments

- YOLOv8 by Ultralytics
- Triplet Loss formulation by Schroff et al. (FaceNet)
- Thermal dataset preprocessing inspired by UAV thermal imaging research

## Contact

For questions or issues, please open an issue on GitHub.
