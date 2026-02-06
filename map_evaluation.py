"""
Mean Average Precision (mAP) Evaluation Module
Implements comprehensive mAP calculation for thermal object detection
"""

import numpy as np
from collections import defaultdict
import json
from pathlib import Path


class mAPEvaluator:
    """
    Calculate mean Average Precision (mAP) for object detection.
    Supports multiple IoU thresholds and COCO-style metrics.
    """
    
    def __init__(self, num_classes, class_names, iou_thresholds=None):
        """
        Initialize mAP evaluator.
        
        Args:
            num_classes (int): Number of object classes
            class_names (list): List of class names
            iou_thresholds (list): IoU thresholds for evaluation
        """
        self.num_classes = num_classes
        self.class_names = class_names
        
        if iou_thresholds is None:
            self.iou_thresholds = np.linspace(0.5, 0.95, 10)
        else:
            self.iou_thresholds = np.array(iou_thresholds)
        
        self.reset()
    
    def reset(self):
        """Reset all stored predictions and ground truths."""
        self.predictions = defaultdict(list)  # class_id -> list of predictions
        self.ground_truths = defaultdict(list)  # class_id -> list of ground truths
        self.image_ids = set()
    
    def add_predictions(self, image_id, pred_boxes, pred_classes, pred_scores):
        """
        Add predictions for an image.
        
        Args:
            image_id: Unique image identifier
            pred_boxes (np.ndarray): Predicted boxes [N, 4] in format [x1, y1, x2, y2]
            pred_classes (np.ndarray): Predicted class IDs [N]
            pred_scores (np.ndarray): Prediction confidence scores [N]
        """
        self.image_ids.add(image_id)
        
        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            self.predictions[int(cls)].append({
                'image_id': image_id,
                'bbox': box,
                'score': score
            })
    
    def add_ground_truths(self, image_id, gt_boxes, gt_classes):
        """
        Add ground truth annotations for an image.
        
        Args:
            image_id: Unique image identifier
            gt_boxes (np.ndarray): Ground truth boxes [N, 4] in format [x1, y1, x2, y2]
            gt_classes (np.ndarray): Ground truth class IDs [N]
        """
        self.image_ids.add(image_id)
        
        for box, cls in zip(gt_boxes, gt_classes):
            self.ground_truths[int(cls)].append({
                'image_id': image_id,
                'bbox': box,
                'difficult': False  # Can be extended to handle difficult samples
            })
    
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) between two boxes.
        
        Args:
            box1, box2 (array-like): Boxes in format [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        # Determine coordinates of intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        return iou
    
    def calculate_ap(self, recalls, precisions):
        """
        Calculate Average Precision (AP) using 11-point interpolation.
        
        Args:
            recalls (np.ndarray): Recall values
            precisions (np.ndarray): Precision values
            
        Returns:
            float: Average Precision
        """
        # Add sentinel values at the beginning and end
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))
        
        # Compute envelope
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for recall_threshold in np.linspace(0, 1, 11):
            # Find precisions at this recall threshold
            indices = np.where(recalls >= recall_threshold)[0]
            if len(indices) > 0:
                ap += precisions[indices[0]]
        
        ap /= 11.0
        
        return ap
    
    def calculate_ap_per_class(self, class_id, iou_threshold=0.5):
        """
        Calculate AP for a specific class at given IoU threshold.
        
        Args:
            class_id (int): Class ID
            iou_threshold (float): IoU threshold for matching
            
        Returns:
            float: Average Precision for the class
        """
        # Get predictions and ground truths for this class
        preds = self.predictions.get(class_id, [])
        gts = self.ground_truths.get(class_id, [])
        
        if len(gts) == 0:
            return 0.0
        
        if len(preds) == 0:
            return 0.0
        
        # Sort predictions by confidence score (descending)
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)
        
        # Group ground truths by image
        gt_by_image = defaultdict(list)
        for gt in gts:
            gt_by_image[gt['image_id']].append(gt)
        
        # Track which ground truths have been matched
        gt_matched = {gt['image_id']: [False] * len(gt_by_image[gt['image_id']]) 
                      for gt in gts}
        
        # Calculate TP and FP for each prediction
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        for pred_idx, pred in enumerate(preds):
            image_id = pred['image_id']
            pred_box = pred['bbox']
            
            # Get ground truths for this image
            image_gts = gt_by_image.get(image_id, [])
            
            if len(image_gts) == 0:
                fp[pred_idx] = 1
                continue
            
            # Find best matching ground truth
            max_iou = 0
            max_gt_idx = -1
            
            for gt_idx, gt in enumerate(image_gts):
                iou = self.calculate_iou(pred_box, gt['bbox'])
                if iou > max_iou:
                    max_iou = iou
                    max_gt_idx = gt_idx
            
            # Check if match is good enough and not already matched
            if max_iou >= iou_threshold:
                if not gt_matched[image_id][max_gt_idx]:
                    tp[pred_idx] = 1
                    gt_matched[image_id][max_gt_idx] = True
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Calculate cumulative TP and FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calculate precision and recall
        total_gts = len(gts)
        recalls = tp_cumsum / total_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Calculate AP
        ap = self.calculate_ap(recalls, precisions)
        
        return ap
    
    def evaluate(self):
        """
        Evaluate mAP across all classes and IoU thresholds.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        results = {
            'per_class': {},
            'mAP@0.5': 0.0,
            'mAP@0.75': 0.0,
            'mAP@0.5:0.95': 0.0
        }
        
        # Calculate AP for each class at each IoU threshold
        aps = np.zeros((self.num_classes, len(self.iou_thresholds)))
        
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            results['per_class'][class_name] = {}
            
            for iou_idx, iou_threshold in enumerate(self.iou_thresholds):
                ap = self.calculate_ap_per_class(class_id, iou_threshold)
                aps[class_id, iou_idx] = ap
                
                # Store specific IoU thresholds
                if abs(iou_threshold - 0.5) < 0.01:
                    results['per_class'][class_name]['AP@0.5'] = ap
                elif abs(iou_threshold - 0.75) < 0.01:
                    results['per_class'][class_name]['AP@0.75'] = ap
            
            # Calculate mAP@0.5:0.95 for this class
            results['per_class'][class_name]['AP@0.5:0.95'] = np.mean(aps[class_id, :])
        
        # Calculate overall mAP metrics
        results['mAP@0.5'] = np.mean(aps[:, 0])  # Assuming first threshold is 0.5
        
        # Find index closest to 0.75
        iou_75_idx = np.argmin(np.abs(self.iou_thresholds - 0.75))
        results['mAP@0.75'] = np.mean(aps[:, iou_75_idx])
        
        # Calculate mAP@0.5:0.95 (mean across all IoU thresholds)
        results['mAP@0.5:0.95'] = np.mean(aps)
        
        # Additional metrics
        results['num_images'] = len(self.image_ids)
        results['num_predictions'] = sum(len(preds) for preds in self.predictions.values())
        results['num_ground_truths'] = sum(len(gts) for gts in self.ground_truths.values())
        
        return results
    
    def print_results(self, results):
        """
        Print evaluation results in a formatted table.
        
        Args:
            results (dict): Evaluation results from evaluate()
        """
        print("\n" + "="*70)
        print("Object Detection Evaluation Results")
        print("="*70)
        
        print(f"\nDataset Statistics:")
        print(f"  Total Images: {results['num_images']}")
        print(f"  Total Predictions: {results['num_predictions']}")
        print(f"  Total Ground Truths: {results['num_ground_truths']}")
        
        print(f"\nOverall mAP:")
        print(f"  mAP@0.5      : {results['mAP@0.5']:.4f}")
        print(f"  mAP@0.75     : {results['mAP@0.75']:.4f}")
        print(f"  mAP@0.5:0.95 : {results['mAP@0.5:0.95']:.4f}")
        
        print(f"\nPer-Class Results:")
        print(f"{'Class':<20} {'AP@0.5':<12} {'AP@0.75':<12} {'AP@0.5:0.95':<12}")
        print("-"*70)
        
        for class_name, metrics in results['per_class'].items():
            ap50 = metrics.get('AP@0.5', 0.0)
            ap75 = metrics.get('AP@0.75', 0.0)
            ap50_95 = metrics.get('AP@0.5:0.95', 0.0)
            print(f"{class_name:<20} {ap50:<12.4f} {ap75:<12.4f} {ap50_95:<12.4f}")
        
        print("="*70 + "\n")
    
    def save_results(self, results, output_path):
        """
        Save evaluation results to JSON file.
        
        Args:
            results (dict): Evaluation results
            output_path (str or Path): Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to: {output_path}")
