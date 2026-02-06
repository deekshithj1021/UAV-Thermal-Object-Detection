# Quick Start Guide

## For Assignment 1: YOLOv8 Thermal Object Detection

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset

Create a YOLO format dataset:
```
data/thermal/
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

### Step 3: Configure Dataset

Copy and edit `data_template.yaml`:
```bash
cp data_template.yaml data.yaml
# Edit data.yaml with your dataset paths and classes
```

### Step 4: Train Model

```python
from thermal_detector import ThermalObjectDetector

detector = ThermalObjectDetector('config.yaml')
detector.load_model('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.
detector.train('data.yaml', epochs=100, batch_size=16)
```

### Step 5: Run Inference

```python
# Detect objects in a single image
detections = detector.detect('path/to/thermal_image.jpg', visualize=True)

# Print results
print(f"Detected {len(detections['boxes'])} objects:")
for cls_name, conf in zip(detections['class_names'], detections['confidences']):
    print(f"  - {cls_name}: {conf:.3f}")
```

### Step 6: Evaluate Model

```python
results = detector.evaluate(
    test_data_path='data/thermal/test/images',
    ground_truth_path='data/thermal/test/labels'
)

# Results include:
# - mAP@0.5
# - mAP@0.75
# - mAP@0.5:0.95 (COCO metric)
# - Per-class AP
```

### Features Implemented

✅ **Data Augmentation**:
- Histogram equalization (CLAHE)
- Thermal drift simulation
- Geometric transformations (rotation, scaling, translation)
- Atmospheric effects (noise, blur, fog)
- Motion blur (UAV simulation)
- Mosaic and MixUp augmentations

✅ **mAP Evaluation**:
- mAP@0.5 (PASCAL VOC)
- mAP@0.75 (stricter threshold)
- mAP@0.5:0.95 (COCO standard)
- Per-class metrics
- Confusion matrix support
- JSON export

✅ **Pipeline Features**:
- YOLOv8 integration
- Configurable via YAML
- Thermal image preprocessing
- Result visualization
- Production-ready code

---

## For Assignment 2: Triplet Loss Mathematical Analysis

### Access the Mathematical Proofs

Read `TRIPLET_LOSS_ANALYSIS.md` for complete derivations:

```bash
# View in terminal
cat TRIPLET_LOSS_ANALYSIS.md

# Or open in editor
nano TRIPLET_LOSS_ANALYSIS.md
```

### Key Results Proven

#### 1. Effect of Margin on Inter-Class Distances

**Theorem 1**: The margin $m$ establishes a lower bound on inter-class distance:
$$d_{\text{inter-class}} \geq d_{\text{intra-class}} + m$$

**Practical meaning**: 
- Larger margin → Better class separation
- Minimum inter-class distance ≥ margin $m$

#### 2. OOD Sample Positioning

**Theorem 3**: OOD samples reside at distance:
$$d_{\text{OOD}} > \max_i \sigma_i + \frac{m}{2}$$

from nearest class prototypes.

**Practical meaning**:
- OOD samples are naturally distant from all classes
- Distance-based detection is theoretically grounded
- Threshold can be set based on intra-class variance + margin

### Topics Covered

1. ✅ Triplet Loss definition and formulation
2. ✅ Mathematical proof of margin effect on inter-class distances
3. ✅ Derivation of inter-class distance bounds
4. ✅ Proof that OOD samples reside further from class prototypes
5. ✅ Quantitative analysis with bounds
6. ✅ Geometric interpretations
7. ✅ Practical OOD detection algorithm
8. ✅ False positive/true positive rate analysis
9. ✅ Connection to metric learning theory

---

## Validation

Run the validation script to verify everything is set up correctly:

```bash
python validate.py
```

Expected output: "All validations PASSED"

---

## File Structure

```
.
├── thermal_detector.py           # Main YOLOv8 pipeline
├── thermal_augmentation.py       # Thermal augmentations
├── map_evaluation.py             # mAP evaluation
├── config.yaml                   # Configuration
├── data_template.yaml            # Dataset template
├── requirements.txt              # Dependencies
├── example_usage.py              # Usage examples
├── validate.py                   # Validation script
│
├── TRIPLET_LOSS_ANALYSIS.md     # Assignment 2: Mathematical proofs
├── ASSIGNMENT_SOLUTIONS.md       # Complete documentation
├── QUICKSTART.md                 # This file
└── README.md                     # Project overview
```

---

## Examples

### Run All Examples
```bash
python example_usage.py
```

This demonstrates:
1. Training workflow
2. Object detection
3. Model evaluation
4. Data augmentation
5. Dataset preparation

### Individual Examples

```python
# Example 1: Load and configure detector
from thermal_detector import ThermalObjectDetector
detector = ThermalObjectDetector('config.yaml')

# Example 2: Apply thermal augmentation
from thermal_augmentation import thermal_histogram_equalization
import cv2
image = cv2.imread('thermal.jpg')
enhanced = thermal_histogram_equalization(image)

# Example 3: Initialize mAP evaluator
from map_evaluation import mAPEvaluator
evaluator = mAPEvaluator(
    num_classes=5,
    class_names=['person', 'car', 'bicycle', 'motorbike', 'bus']
)
```

---

## Troubleshooting

### Issue: Dependencies not installed
```bash
pip install -r requirements.txt
```

### Issue: CUDA not available
Edit `config.yaml`:
```yaml
training:
  device: 'cpu'  # Change from 'cuda' to 'cpu'
```

### Issue: Dataset format
Ensure labels are in YOLO format:
```
class_id x_center y_center width height
```
All values normalized [0, 1]

---

## Documentation

- **README.md**: Project overview and features
- **TRIPLET_LOSS_ANALYSIS.md**: Complete mathematical proofs (Assignment 2)
- **ASSIGNMENT_SOLUTIONS.md**: Detailed implementation documentation
- **QUICKSTART.md**: This quick start guide (you are here)

---

## Support

For issues or questions:
1. Check the documentation files
2. Review example_usage.py for code examples
3. Run validate.py to check setup
4. Open an issue on GitHub

---

## Citation

If you use this code, please cite:

```bibtex
@misc{uav-thermal-detection-2024,
  title={UAV Thermal Object Detection with Triplet Loss Analysis},
  author={Deekshith J},
  year={2024},
  howpublished={\url{https://github.com/deekshithj1021/UAV-Thermal-Object-Detection}}
}
```

---

**Both assignments are complete and ready to use!**

✓ Assignment 1: YOLOv8 pipeline with thermal augmentations and mAP  
✓ Assignment 2: Mathematical proofs for Triplet Loss and OOD detection
