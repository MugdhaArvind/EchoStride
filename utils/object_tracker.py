import numpy as np
import cv2
from models.model_utils import get_object_distances
from config import MAX_OBJECTS

class ObjectTracker:
    def __init__(self):
        # We'll use a simple tracking approach based on IOU (Intersection over Union)
        self.prev_detections = []
        self.tracking_id = 0
        
    def update(self, detections, frame_width, frame_height):
        """
        Update tracking information for detected objects
        Returns: enhanced detections with positions and tracking IDs
        """
        # Get positional information for each detection
        enhanced_detections = []
        
        # Sort by confidence
        detections.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to maximum number of objects to avoid information overload
        detections = detections[:MAX_OBJECTS]
        
        # Calculate positions
        for label, confidence, box in detections:
            positions = get_object_distances([box], frame_width, frame_height)[0]
            enhanced_detections.append((label, confidence, box, positions))
            
        # Return the enhanced detections
        return enhanced_detections
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union for two boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate area of intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both boxes
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou