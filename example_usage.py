"""
Example Usage Script for Thermal Object Detection Pipeline
Demonstrates training, inference, and evaluation
"""

import os
from pathlib import Path
from thermal_detector import ThermalObjectDetector


def example_training():
    """Example: Train YOLOv8 on thermal dataset."""
    print("\n" + "="*70)
    print("Example 1: Training YOLOv8 on Thermal Imagery")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = ThermalObjectDetector('config.yaml')
    
    # Load pretrained YOLOv8 model
    detector.load_model('yolov8n.pt')  # nano model for faster training
    
    # Train on thermal dataset
    # Note: You need to prepare a data.yaml file with your dataset paths
    print("To train the model, prepare a data.yaml file with:")
    print("""
    path: /path/to/dataset
    train: train/images
    val: val/images
    test: test/images
    
    nc: 5  # number of classes
    names: ['person', 'car', 'bicycle', 'motorbike', 'bus']
    """)
    
    # Uncomment to run actual training:
    # detector.train(
    #     data_yaml_path='data.yaml',
    #     epochs=100,
    #     batch_size=16
    # )
    
    print("\nTraining command: detector.train('data.yaml', epochs=100)")


def example_inference():
    """Example: Detect objects in thermal images."""
    print("\n" + "="*70)
    print("Example 2: Object Detection in Thermal Images")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = ThermalObjectDetector('config.yaml')
    
    # Load trained model
    # detector.load_model('runs/detect/thermal_detection/weights/best.pt')
    detector.load_model('yolov8n.pt')  # For demonstration
    
    print("To detect objects in a thermal image:")
    print("detections = detector.detect('path/to/thermal_image.jpg', visualize=True)")
    print("\nThis will:")
    print("  1. Preprocess the thermal image")
    print("  2. Apply histogram equalization")
    print("  3. Run YOLOv8 inference")
    print("  4. Visualize results with bounding boxes")
    print("  5. Save visualization to runs/detect/visualizations/")
    
    # Example detection (uncomment with actual image):
    # detections = detector.detect(
    #     'examples/thermal_image.jpg',
    #     conf_threshold=0.25,
    #     iou_threshold=0.45,
    #     visualize=True
    # )
    # 
    # print(f"\nDetected {len(detections['boxes'])} objects:")
    # for cls_name, conf in zip(detections['class_names'], detections['confidences']):
    #     print(f"  - {cls_name}: {conf:.2f}")


def example_evaluation():
    """Example: Evaluate model and calculate mAP."""
    print("\n" + "="*70)
    print("Example 3: Model Evaluation with mAP")
    print("="*70 + "\n")
    
    # Initialize detector
    detector = ThermalObjectDetector('config.yaml')
    
    # Load trained model
    detector.load_model('yolov8n.pt')  # Use your trained model
    
    print("To evaluate the model on test data:")
    print("""
    results = detector.evaluate(
        test_data_path='data/thermal/test/images',
        ground_truth_path='data/thermal/test/labels'
    )
    """)
    
    print("\nThis will calculate:")
    print("  - mAP@0.5      : Mean Average Precision at IoU=0.5")
    print("  - mAP@0.75     : Mean Average Precision at IoU=0.75")
    print("  - mAP@0.5:0.95 : COCO-style mAP (average across IoU thresholds)")
    print("  - Per-class AP : Average Precision for each class")
    
    # Example evaluation (uncomment with actual data):
    # results = detector.evaluate(
    #     test_data_path='data/thermal/test/images',
    #     ground_truth_path='data/thermal/test/labels'
    # )


def example_augmentation():
    """Example: Demonstrate thermal-specific augmentations."""
    print("\n" + "="*70)
    print("Example 4: Thermal Image Augmentation")
    print("="*70 + "\n")
    
    import cv2
    import numpy as np
    from thermal_augmentation import (
        thermal_histogram_equalization,
        simulate_thermal_drift,
        apply_mosaic_augmentation
    )
    
    print("Thermal-specific augmentations implemented:")
    print("\n1. Histogram Equalization (CLAHE):")
    print("   - Enhances contrast in thermal images")
    print("   - Useful for low-contrast thermal scenes")
    
    print("\n2. Thermal Drift Simulation:")
    print("   - Simulates gradual temperature changes")
    print("   - Models UAV camera sensor drift")
    
    print("\n3. Geometric Augmentations:")
    print("   - Rotation, translation, scaling")
    print("   - Flips (horizontal/vertical)")
    print("   - Perspective transformation")
    
    print("\n4. Atmospheric Effects:")
    print("   - Gaussian noise (sensor noise)")
    print("   - Gaussian blur (atmospheric scattering)")
    print("   - Motion blur (UAV movement)")
    print("   - Random fog")
    
    print("\n5. Advanced Augmentations:")
    print("   - Mosaic (combines 4 images)")
    print("   - MixUp (blends 2 images)")
    print("   - Gamma adjustment")
    
    # Example usage (uncomment with actual image):
    # image = cv2.imread('thermal_image.jpg')
    # 
    # # Apply histogram equalization
    # enhanced = thermal_histogram_equalization(image)
    # 
    # # Apply thermal drift
    # drifted = simulate_thermal_drift(image, drift_amount=0.1)


def example_custom_dataset():
    """Example: Prepare custom thermal dataset."""
    print("\n" + "="*70)
    print("Example 5: Preparing Custom Thermal Dataset")
    print("="*70 + "\n")
    
    print("Dataset Structure (YOLO format):")
    print("""
    dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   └── labels/
    │       ├── image1.txt
    │       ├── image2.txt
    │       └── ...
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
    """)
    
    print("\nLabel Format (YOLO): class_id x_center y_center width height")
    print("Example: 0 0.5 0.5 0.3 0.4")
    print("  - class_id: 0 (person)")
    print("  - x_center: 0.5 (center at 50% of image width)")
    print("  - y_center: 0.5 (center at 50% of image height)")
    print("  - width: 0.3 (30% of image width)")
    print("  - height: 0.4 (40% of image height)")
    
    print("\ndata.yaml:")
    print("""
    path: ./dataset
    train: train/images
    val: val/images
    test: test/images
    
    nc: 5
    names: ['person', 'car', 'bicycle', 'motorbike', 'bus']
    """)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("YOLOv8 Thermal Object Detection Pipeline - Examples")
    print("="*70)
    
    examples = [
        ("Training", example_training),
        ("Inference", example_inference),
        ("Evaluation", example_evaluation),
        ("Augmentation", example_augmentation),
        ("Custom Dataset", example_custom_dataset)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "-"*70)
    
    # Run all examples
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
    
    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")
    
    print("Next Steps:")
    print("1. Prepare your thermal imagery dataset")
    print("2. Create data.yaml configuration file")
    print("3. Train the model: python thermal_detector.py")
    print("4. Evaluate results: see evaluation_results.json")
    print("\nFor mathematical analysis of Triplet Loss and OOD detection,")
    print("see: TRIPLET_LOSS_ANALYSIS.md")


if __name__ == "__main__":
    main()
