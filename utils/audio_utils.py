import pyttsx3
import pygame
import time
import threading
from queue import Queue, Empty
from config import VOICE_RATE, VOICE_ID, VOLUME

class AudioFeedback:
    def __init__(self):
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', VOICE_RATE)
        if VOICE_ID:
            self.engine.setProperty('voice', VOICE_ID)
            
        # Initialize pygame for sound effects
        pygame.mixer.init()
        
        # Message queue and speaking state
        self.message_queue = Queue()
        self.is_speaking = False
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        
        # Cooldown tracking for object announcements
        self.last_announcements = {}
        self.cooldown = 3  # seconds between repeated announcements
        
    def _process_queue(self):
        """Worker thread to process audio messages"""
        while True:
            try:
                message = self.message_queue.get(timeout=0.1)
                self.is_speaking = True
                self.engine.say(message)
                self.engine.runAndWait()
                self.is_speaking = False
                self.message_queue.task_done()
            except Empty:
                time.sleep(0.1)
                
    def say(self, message, priority=False):
        """Add message to the queue to be spoken"""
        if priority:
            # Clear existing messages for priority announcements
            while not self.message_queue.empty():
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.task_done()
                except Empty:
                    break
        
        self.message_queue.put(message)
        
    def announce_objects(self, detections, image_width, image_height):
        """
        Create spoken announcements for detected objects
        detections: list of (label, confidence, box, position)
        """
        current_time = time.time()
        announcements = []
        
        for label, _, box, (horizontal, distance) in detections:
            # Create a key for this object type and position
            key = f"{label}_{horizontal}_{distance}"
            
            # Check if we've announced this object recently
            if key in self.last_announcements and current_time - self.last_announcements[key] < self.cooldown:
                continue
                
            # Create announcement message
            if horizontal == "center" and distance == "very close":
                message = f"{label} directly ahead, very close"
            else:
                message = f"{label} to the {horizontal}, {distance}"
                
            announcements.append(message)
            self.last_announcements[key] = current_time
            
        # Combine announcements
        if announcements:
            combined_message = ". ".join(announcements)
            self.say(combined_message)
            
    def announce_system_status(self, message):
        """Announce system status messages"""
        self.say(message, priority=True)