import cv2
import time
import numpy as np
import torch
import threading
from models.ssd_detector import SSDDetector
from models.dvt_classifier import DynamicVisionTransformer
from utils.camera_utils import Camera
from utils.audio_utils import AudioFeedback
from utils.object_tracker import ObjectTracker
from config import DETECTION_FREQUENCY, FRAME_WIDTH, FRAME_HEIGHT

class EchoStride:
    def __init__(self):
        print("Initializing EchoStride...")
        print("Loading models... This may take a few moments.")
        
        # Initialize components
        self.camera = Camera()
        self.audio = AudioFeedback()
        self.detector = SSDDetector()
        self.transformer = DynamicVisionTransformer()
        self.tracker = ObjectTracker()
        
        # System state
        self.running = False
        self.last_detection_time = 0
        
        print("EchoStride initialized and ready!")
        self.audio.announce_system_status("Echo Stride ready")
        
    def process_frame(self, frame):
        """Process a single frame from the camera"""
        current_time = time.time()
        
        # Perform detection at specified intervals
        if current_time - self.last_detection_time >= DETECTION_FREQUENCY:
            # Detect objects using SSD
            detections = self.detector.detect(frame)
            
            # Track and enhance detections with positional info
            enhanced_detections = self.tracker.update(detections, FRAME_WIDTH, FRAME_HEIGHT)
            
            # Announce detected objects
            self.audio.announce_objects(enhanced_detections, FRAME_WIDTH, FRAME_HEIGHT)
            
            self.last_detection_time = current_time
            
            # Visualize detections (for debugging purposes)
            for label, confidence, box, (position, distance) in enhanced_detections:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        return frame
            
    def run(self):
        """Main loop for the EchoStride system"""
        self.running = True
        self.audio.announce_system_status("Starting object detection")
        
        try:
            while self.running:
                # Get frame from camera
                frame = self.camera.get_frame()
                if frame is None:
                    continue
                    
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display frame (for development/debugging)
                cv2.imshow("EchoStride", processed_frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Stopping EchoStride...")
        finally:
            self.running = False
            self.camera.release()
            cv2.destroyAllWindows()
            self.audio.announce_system_status("Echo Stride shutting down")
            
    def stop(self):
        """Stop the system"""
        self.running = False

if __name__ == "__main__":
    # Create and run the EchoStride system
    echo_stride = EchoStride()
    echo_stride.run()