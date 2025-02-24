import torch
from transformers import DeiTFeatureExtractor, DeiTForImageClassification
import cv2
import numpy as np
from models.model_utils import get_device

class DynamicVisionTransformer:
    def __init__(self):
        self.device = get_device()
        
        # We'll use DeiT (Data-efficient image Transformer) as our transformer model
        # It's more lightweight than ViT and works well for real-time applications
        self.feature_extractor = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-patch16-224')
        self.model = DeiTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')
        self.model.to(self.device)
        self.model.eval()
        
    def classify(self, image, boxes):
        """
        Classify detected objects using the transformer model
        This can provide additional context or verification for the SSD detections
        """
        results = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Extract the object region
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            
            obj_img = image[y1:y2, x1:x2]
            
            if obj_img.size == 0:
                continue
                
            # Resize to fit transformer input
            obj_img = cv2.resize(obj_img, (224, 224))
            
            # Prepare input for transformer
            inputs = self.feature_extractor(images=obj_img, return_tensors="pt").to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get predictions
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            
            # Map to ImageNet class
            confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()
            
            results.append((predicted_class_idx, confidence))
            
        return results