import cv2
import time
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, FPS

class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        if not self.cap.isOpened():
            raise ValueError("Could not open camera. Check if it's connected properly.")
        
        # Warm up the camera
        for _ in range(10):
            ret, _ = self.cap.read()
            if not ret:
                print("Warning: Failed to grab initial frames.")
        
    def get_frame(self):
        """Get a frame from the camera"""
        ret, frame = self.cap.read()
        if not ret:
            print("Warning: Failed to grab frame.")
            return None
        return frame
    
    def release(self):
        """Release the camera resources"""
        self.cap.release()