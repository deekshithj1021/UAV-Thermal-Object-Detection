"""
YOLOv8 Thermal Object Detection Pipeline
Main script for training and inference on thermal imagery
"""

import os
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

from thermal_augmentation import (
    ThermalAugmentation,
    apply_mosaic_augmentation,
    apply_mixup_augmentation,
    thermal_histogram_equalization
)
from map_evaluation import mAPEvaluator


class ThermalObjectDetector:
    """
    YOLOv8-based thermal object detector with custom augmentations.
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Initialize thermal object detector.
        
        Args:
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config['training']['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = 'cpu'
        
        self.model = None
        self.augmentation = ThermalAugmentation(self.config['augmentation'])
    
    def load_model(self, model_path=None):
        """
        Load YOLOv8 model.
        
        Args:
            model_path (str): Path to model weights. If None, loads pretrained model.
        """
        if model_path is None:
            model_path = self.config['model']['name']
        
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        return self.model
    
    def preprocess_thermal_image(self, image_path, apply_histogram_eq=True):
        """
        Preprocess thermal image for detection.
        
        Args:
            image_path (str): Path to thermal image
            apply_histogram_eq (bool): Whether to apply histogram equalization
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Read image
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to RGB (YOLOv8 expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply histogram equalization for better thermal contrast
        if apply_histogram_eq:
            image = thermal_histogram_equalization(image)
        
        return image
    
    def train(self, data_yaml_path, epochs=None, batch_size=None):
        """
        Train YOLOv8 model on thermal dataset.
        
        Args:
            data_yaml_path (str): Path to dataset YAML configuration
            epochs (int): Number of training epochs (overrides config)
            batch_size (int): Batch size (overrides config)
        """
        if self.model is None:
            self.load_model()
        
        # Training parameters
        epochs = epochs or self.config['training']['epochs']
        batch_size = batch_size or self.config['training']['batch_size']
        img_size = self.config['model']['input_size']
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {img_size}")
        print(f"Device: {self.device}")
        
        # Train model
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=self.device,
            workers=self.config['training']['workers'],
            optimizer='Adam',
            lr0=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            momentum=self.config['training']['momentum'],
            warmup_epochs=self.config['training']['warmup_epochs'],
            save_period=self.config['training']['save_period'],
            
            # Data augmentation settings
            hsv_h=self.config['augmentation']['hsv_h'],
            hsv_s=self.config['augmentation']['hsv_s'],
            hsv_v=self.config['augmentation']['hsv_v'],
            degrees=self.config['augmentation']['degrees'],
            translate=self.config['augmentation']['translate'],
            scale=self.config['augmentation']['scale'],
            shear=self.config['augmentation']['shear'],
            perspective=self.config['augmentation']['perspective'],
            flipud=self.config['augmentation']['flipud'],
            fliplr=self.config['augmentation']['fliplr'],
            mosaic=self.config['augmentation']['mosaic'],
            mixup=self.config['augmentation']['mixup'],
            
            # Output settings
            project=self.config['output']['save_dir'],
            name='thermal_detection',
            exist_ok=True,
            save_txt=self.config['output']['save_txt'],
            save_conf=self.config['output']['save_conf'],
            save_json=self.config['output']['save_json'],
        )
        
        print("Training completed!")
        return results
    
    def detect(self, image_path, conf_threshold=None, iou_threshold=None, visualize=True):
        """
        Detect objects in thermal image.
        
        Args:
            image_path (str): Path to thermal image
            conf_threshold (float): Confidence threshold
            iou_threshold (float): IoU threshold for NMS
            visualize (bool): Whether to visualize results
            
        Returns:
            dict: Detection results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Get thresholds from config if not provided
        conf_threshold = conf_threshold or self.config['model']['confidence_threshold']
        iou_threshold = iou_threshold or self.config['model']['iou_threshold']
        
        # Preprocess image
        image = self.preprocess_thermal_image(image_path)
        
        # Run inference
        results = self.model.predict(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = {
            'boxes': [],
            'classes': [],
            'confidences': [],
            'class_names': []
        }
        
        if len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                detections['boxes'] = boxes
                detections['classes'] = classes
                detections['confidences'] = confidences
                detections['class_names'] = [
                    self.model.names[cls] for cls in classes
                ]
        
        # Visualize if requested
        if visualize and len(detections['boxes']) > 0:
            self.visualize_detections(image, detections, image_path)
        
        return detections
    
    def visualize_detections(self, image, detections, image_path):
        """
        Visualize detection results.
        
        Args:
            image (np.ndarray): Input image
            detections (dict): Detection results
            image_path (str): Original image path for saving
        """
        img_vis = image.copy()
        
        # Draw bounding boxes
        for box, cls_name, conf in zip(
            detections['boxes'],
            detections['class_names'],
            detections['confidences']
        ):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                img_vis,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            cv2.putText(
                img_vis,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        # Save visualization
        output_dir = Path(self.config['output']['save_dir']) / 'visualizations'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / Path(image_path).name
        img_vis_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), img_vis_bgr)
        
        print(f"Visualization saved to: {output_path}")
    
    def evaluate(self, test_data_path, ground_truth_path=None):
        """
        Evaluate model on test dataset and calculate mAP.
        
        Args:
            test_data_path (str): Path to test images directory
            ground_truth_path (str): Path to ground truth annotations
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print("Evaluating model...")
        
        # Initialize mAP evaluator
        num_classes = self.config['dataset']['num_classes']
        class_names = self.config['dataset']['class_names']
        iou_thresholds = self.config['evaluation']['map_iou_thresholds']
        
        evaluator = mAPEvaluator(num_classes, class_names, iou_thresholds)
        
        # Get test images
        test_path = Path(test_data_path)
        image_files = list(test_path.glob('*.jpg')) + list(test_path.glob('*.png'))
        
        print(f"Found {len(image_files)} test images")
        
        # Process each image
        for img_idx, image_path in enumerate(tqdm(image_files, desc="Evaluating")):
            # Get predictions
            detections = self.detect(
                image_path,
                visualize=self.config['output']['visualize']
            )
            
            # Add predictions to evaluator
            if len(detections['boxes']) > 0:
                evaluator.add_predictions(
                    image_id=img_idx,
                    pred_boxes=detections['boxes'],
                    pred_classes=detections['classes'],
                    pred_scores=detections['confidences']
                )
            
            # Load and add ground truth
            if ground_truth_path is not None:
                # Assume YOLO format labels
                label_path = Path(ground_truth_path) / f"{image_path.stem}.txt"
                
                if label_path.exists():
                    gt_boxes, gt_classes = self.load_yolo_labels(
                        label_path,
                        image_path
                    )
                    
                    if len(gt_boxes) > 0:
                        evaluator.add_ground_truths(
                            image_id=img_idx,
                            gt_boxes=gt_boxes,
                            gt_classes=gt_classes
                        )
        
        # Calculate mAP
        results = evaluator.evaluate()
        evaluator.print_results(results)
        
        # Save results
        if self.config['evaluation']['save_predictions']:
            output_path = Path(self.config['output']['save_dir']) / 'evaluation_results.json'
            evaluator.save_results(results, output_path)
        
        return results
    
    def load_yolo_labels(self, label_path, image_path):
        """
        Load YOLO format labels and convert to detection format.
        
        Args:
            label_path (Path): Path to label file
            image_path (Path): Path to corresponding image
            
        Returns:
            tuple: (boxes, classes) in detection format
        """
        # Read image to get dimensions
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        boxes = []
        classes = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert from YOLO format to [x1, y1, x2, y2]
                    x1 = (x_center - width / 2) * w
                    y1 = (y_center - height / 2) * h
                    x2 = (x_center + width / 2) * w
                    y2 = (y_center + height / 2) * h
                    
                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls)
        
        return np.array(boxes), np.array(classes)


def main():
    """Example usage of the thermal object detector."""
    
    # Initialize detector
    detector = ThermalObjectDetector('config.yaml')
    
    # Load model
    detector.load_model()
    
    # Example: Train model (uncomment to use)
    # detector.train('path/to/data.yaml', epochs=100, batch_size=16)
    
    # Example: Detect objects in a single image
    # detections = detector.detect('path/to/thermal/image.jpg', visualize=True)
    # print(f"Detected {len(detections['boxes'])} objects")
    
    # Example: Evaluate model
    # results = detector.evaluate(
    #     test_data_path='data/thermal/test/images',
    #     ground_truth_path='data/thermal/test/labels'
    # )
    
    print("Thermal object detector initialized successfully!")
    print("To use the detector:")
    print("1. Train: detector.train('data.yaml')")
    print("2. Detect: detector.detect('image.jpg')")
    print("3. Evaluate: detector.evaluate('test_dir', 'labels_dir')")


if __name__ == "__main__":
    main()
