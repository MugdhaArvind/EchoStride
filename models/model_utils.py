import torch
import numpy as np
from config import COCO_CLASSES

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_image_for_ssd(image, device):
    # Convert to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
        
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
    return image_tensor.unsqueeze(0).to(device)

def get_object_distances(boxes, image_width, image_height):
    """
    Calculate approximate distance/position of objects based on their bounding boxes
    Returns positions as "left", "right", "center", etc.
    """
    distances = []
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Calculate center point of the box
        center_x = (x1 + x2) / 2
        width = x2 - x1
        
        # Determine horizontal position
        if center_x < image_width * 0.33:
            horizontal = "left"
        elif center_x > image_width * 0.66:
            horizontal = "right"
        else:
            horizontal = "center"
            
        # Calculate approximate size as a rough distance metric
        size_ratio = width / image_width
        if size_ratio > 0.5:
            distance = "very close"
        elif size_ratio > 0.3:
            distance = "close"
        elif size_ratio > 0.1:
            distance = "nearby"
        else:
            distance = "in the distance"
            
        distances.append((horizontal, distance))
        
    return distances