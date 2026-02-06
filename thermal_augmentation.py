"""
Thermal Image Data Augmentation Module
Implements specialized augmentation techniques for thermal imagery
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class ThermalAugmentation:
    """
    Custom augmentation pipeline for thermal imagery.
    Thermal images have unique characteristics that require specialized augmentation.
    """
    
    def __init__(self, config):
        """
        Initialize augmentation pipeline with configuration.
        
        Args:
            config (dict): Augmentation configuration parameters
        """
        self.config = config
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """Build albumentations transform pipeline."""
        transforms = []
        
        if self.config.get('enable', True):
            # Geometric transformations
            if self.config.get('degrees', 0) > 0:
                transforms.append(A.Rotate(
                    limit=self.config['degrees'],
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ))
            
            if self.config.get('translate', 0) > 0:
                translate = self.config['translate']
                transforms.append(A.ShiftScaleRotate(
                    shift_limit=translate,
                    scale_limit=0,
                    rotate_limit=0,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5
                ))
            
            if self.config.get('scale', 0) > 0:
                scale = self.config['scale']
                transforms.append(A.RandomScale(
                    scale_limit=scale,
                    p=0.5
                ))
            
            # Flip transformations
            if self.config.get('fliplr', 0) > 0:
                transforms.append(A.HorizontalFlip(p=self.config['fliplr']))
            
            if self.config.get('flipud', 0) > 0:
                transforms.append(A.VerticalFlip(p=self.config['flipud']))
            
            # Perspective transformation
            if self.config.get('perspective', 0) > 0:
                transforms.append(A.Perspective(
                    scale=self.config['perspective'],
                    p=0.3
                ))
            
            # Thermal-specific augmentations
            # Adjust brightness and contrast (thermal intensity variations)
            if self.config.get('brightness_adjustment', 0) > 0:
                brightness = self.config['brightness_adjustment']
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=brightness,
                    contrast_limit=self.config.get('contrast_adjustment', 0.2),
                    p=0.5
                ))
            
            # Gaussian noise (sensor noise simulation)
            if self.config.get('gaussian_noise', 0) > 0:
                var_limit = self.config['gaussian_noise'] * 255 ** 2
                transforms.append(A.GaussNoise(
                    var_limit=(var_limit / 2, var_limit),
                    p=0.3
                ))
            
            # Gaussian blur (atmospheric effects)
            if self.config.get('gaussian_blur', 0) > 0:
                transforms.append(A.GaussianBlur(
                    blur_limit=(3, 7),
                    p=self.config['gaussian_blur']
                ))
            
            # Additional thermal-specific augmentations
            transforms.extend([
                # Simulate different thermal conditions
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                
                # Simulate atmospheric scattering
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
                
                # Motion blur (UAV movement)
                A.MotionBlur(blur_limit=7, p=0.2),
            ])
        
        # Normalization
        transforms.append(A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ))
        
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def __call__(self, image, bboxes=None, class_labels=None):
        """
        Apply augmentation to image and bounding boxes.
        
        Args:
            image (np.ndarray): Input image
            bboxes (list): List of bounding boxes in YOLO format [x_center, y_center, width, height]
            class_labels (list): List of class labels for each bbox
            
        Returns:
            dict: Augmented image and bounding boxes
        """
        if bboxes is None:
            bboxes = []
        if class_labels is None:
            class_labels = []
        
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        return transformed


def apply_mosaic_augmentation(images, labels, img_size=640, prob=1.0):
    """
    Apply mosaic augmentation - combines 4 images into one.
    Effective for small object detection in thermal imagery.
    
    Args:
        images (list): List of 4 images
        labels (list): List of 4 label sets
        img_size (int): Output image size
        prob (float): Probability of applying mosaic
        
    Returns:
        tuple: Augmented image and labels
    """
    if random.random() > prob or len(images) < 4:
        return images[0], labels[0]
    
    # Create output image
    mosaic_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    mosaic_labels = []
    
    # Divide image into 4 quadrants
    yc, xc = img_size // 2, img_size // 2
    
    for i, (img, label) in enumerate(zip(images[:4], labels[:4])):
        h, w = img.shape[:2]
        
        # Determine position in mosaic
        if i == 0:  # top-left
            x1, y1, x2, y2 = 0, 0, xc, yc
        elif i == 1:  # top-right
            x1, y1, x2, y2 = xc, 0, img_size, yc
        elif i == 2:  # bottom-left
            x1, y1, x2, y2 = 0, yc, xc, img_size
        else:  # bottom-right
            x1, y1, x2, y2 = xc, yc, img_size, img_size
        
        # Resize and place image
        img_resized = cv2.resize(img, (x2 - x1, y2 - y1))
        mosaic_img[y1:y2, x1:x2] = img_resized
        
        # Adjust labels
        for bbox in label:
            cls, x_center, y_center, width, height = bbox
            
            # Scale to mosaic coordinates
            new_x = x1 + x_center * (x2 - x1) / img_size
            new_y = y1 + y_center * (y2 - y1) / img_size
            new_w = width * (x2 - x1) / img_size
            new_h = height * (y2 - y1) / img_size
            
            mosaic_labels.append([cls, new_x, new_y, new_w, new_h])
    
    return mosaic_img, mosaic_labels


def apply_mixup_augmentation(img1, labels1, img2, labels2, alpha=0.5, prob=0.1):
    """
    Apply MixUp augmentation - blends two images.
    Useful for regularization in thermal object detection.
    
    Args:
        img1, img2 (np.ndarray): Input images
        labels1, labels2 (list): Labels for each image
        alpha (float): Mixing ratio
        prob (float): Probability of applying mixup
        
    Returns:
        tuple: Mixed image and labels
    """
    if random.random() > prob:
        return img1, labels1
    
    # Ensure images have same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Mix images
    mixed_img = (alpha * img1 + (1 - alpha) * img2).astype(np.uint8)
    
    # Combine labels
    mixed_labels = labels1 + labels2
    
    return mixed_img, mixed_labels


def thermal_histogram_equalization(image):
    """
    Apply histogram equalization specifically tuned for thermal imagery.
    Enhances contrast in thermal images.
    
    Args:
        image (np.ndarray): Input thermal image
        
    Returns:
        np.ndarray: Equalized image
    """
    if len(image.shape) == 3:
        # Convert to grayscale for thermal processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    if len(image.shape) == 3:
        # Convert back to color
        equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    return equalized


def simulate_thermal_drift(image, drift_amount=0.1):
    """
    Simulate thermal drift - gradual temperature changes across the image.
    Common in UAV thermal cameras.
    
    Args:
        image (np.ndarray): Input image
        drift_amount (float): Amount of drift to apply
        
    Returns:
        np.ndarray: Image with simulated drift
    """
    h, w = image.shape[:2]
    
    # Create gradient mask
    x_gradient = np.linspace(-drift_amount, drift_amount, w)
    y_gradient = np.linspace(-drift_amount, drift_amount, h)
    xx, yy = np.meshgrid(x_gradient, y_gradient)
    
    drift_mask = (xx + yy) * 255
    drift_mask = drift_mask.astype(np.float32)
    
    # Apply drift
    if len(image.shape) == 3:
        drift_mask = np.stack([drift_mask] * 3, axis=-1)
    
    drifted = np.clip(image.astype(np.float32) + drift_mask, 0, 255).astype(np.uint8)
    
    return drifted
