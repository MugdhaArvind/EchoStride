import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16
from config import CONFIDENCE_THRESHOLD, COCO_CLASSES
from models.model_utils import get_device, prepare_image_for_ssd

class SSDDetector:
    def __init__(self):
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Load pre-trained SSD model
        self.model = ssd300_vgg16(pretrained=True)
        self.model.eval()
        self.model.to(self.device)
        
    def detect(self, image):
        """
        Detect objects in the image using SSD
        Returns: list of (label, confidence, bounding_box)
        """
        # Prepare image for the model
        tensor = prepare_image_for_ssd(image, self.device)
        
        # Perform inference
        with torch.no_grad():
            predictions = self.model(tensor)
            
        # Extract predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        
        # Filter by confidence
        mask = scores >= CONFIDENCE_THRESHOLD
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        results = []
        for i in range(len(boxes)):
            label_idx = labels[i] - 1  # SSD labels start from 1
            if 0 <= label_idx < len(COCO_CLASSES):
                label = COCO_CLASSES[label_idx]
                results.append((label, scores[i], boxes[i]))
                
        return results