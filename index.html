#!/usr/bin/env python3
"""
Golf Grip Hand Measurement Tool
Clean, standalone implementation for testing and Git deployment

Features:
- Real-time hand tracking and measurement
- Automatic hand length and finger length calculation
- Calibration system for accurate measurements
- Simple command-line interface
- Minimal dependencies

Usage:
    python golf_grip_measurement.py

Author: GripScript Team
Version: 1.0.0
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import json
import os
from typing import Dict, Tuple, Optional, List

class GolfGripMeasurement:
    """Clean, standalone golf grip measurement tool with dynamic calibration"""
    
    def __init__(self, average_palm_width_cm: float = 8.5):
        """
        Initialize the measurement tool with dynamic calibration
        
        Args:
            average_palm_width_cm: Average palm width in cm for calibration (default 8.5cm)
        """
        self.average_palm_width_cm = average_palm_width_cm
        self.calibration_factor = 0.05  # Will be calculated dynamically
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Measurement history for stability
        self.measurement_history = []
        self.max_history = 5
        
        # Dynamic calibration tracking
        self.calibration_history = []
        self.max_calibration_history = 10
    
    def calculate_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def pixel_to_cm(self, pixel_distance: float) -> float:
        """Convert pixel distance to centimeters using dynamic calibration"""
        return pixel_distance * self.calibration_factor
    
    def get_palm_width_pixels(self, landmarks: List, frame_width: int, frame_height: int) -> float:
        """
        Calculate palm width in pixels using MediaPipe landmarks
        
        Uses the distance between landmarks 5 (index finger base) and 17 (pinky finger base)
        which represents the palm width across the knuckles
        """
        # Index finger base (landmark 5)
        index_base = landmarks[5]
        index_base_pixel = (int(index_base.x * frame_width), int(index_base.y * frame_height))
        
        # Pinky finger base (landmark 17)
        pinky_base = landmarks[17]
        pinky_base_pixel = (int(pinky_base.x * frame_width), int(pinky_base.y * frame_height))
        
        # Calculate palm width in pixels
        palm_width_pixels = self.calculate_distance(index_base_pixel, pinky_base_pixel)
        return palm_width_pixels
    
    def calculate_dynamic_calibration(self, palm_width_pixels: float) -> float:
        """
        Calculate calibration factor based on palm width
        
        Args:
            palm_width_pixels: Palm width in pixels
            
        Returns:
            Calibration factor (cm/pixel)
        """
        if palm_width_pixels > 0:
            calibration_factor = self.average_palm_width_cm / palm_width_pixels
            return calibration_factor
        return self.calibration_factor
    
    def update_dynamic_calibration(self, palm_width_pixels: float):
        """
        Update calibration factor using palm width detection
        
        Args:
            palm_width_pixels: Palm width in pixels
        """
        if palm_width_pixels > 0:
            new_calibration = self.calculate_dynamic_calibration(palm_width_pixels)
            
            # Add to calibration history for stability
            self.calibration_history.append(new_calibration)
            if len(self.calibration_history) > self.max_calibration_history:
                self.calibration_history.pop(0)
            
            # Use averaged calibration for stability
            if len(self.calibration_history) >= 3:
                self.calibration_factor = sum(self.calibration_history) / len(self.calibration_history)
            else:
                self.calibration_factor = new_calibration
    
    def get_calibration_confidence(self) -> float:
        """Get confidence in current calibration based on history stability"""
        if len(self.calibration_history) < 3:
            return 0.0
        
        # Calculate standard deviation of recent calibrations
        mean_cal = sum(self.calibration_history) / len(self.calibration_history)
        variance = sum((x - mean_cal) ** 2 for x in self.calibration_history) / len(self.calibration_history)
        std_dev = math.sqrt(variance)
        
        # Confidence is inverse of coefficient of variation
        if mean_cal > 0:
            cv = std_dev / mean_cal
            confidence = max(0.0, 1.0 - cv)
            return min(1.0, confidence)
        return 0.0
    
    def get_hand_landmarks(self, frame: np.ndarray) -> Optional[List]:
        """Extract hand landmarks from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0].landmark
        return None
    
    def measure_hand_length(self, landmarks: List, frame_width: int, frame_height: int) -> float:
        """Measure hand length from wrist to middle finger tip"""
        # Wrist (landmark 0)
        wrist = landmarks[0]
        wrist_pixel = (int(wrist.x * frame_width), int(wrist.y * frame_height))
        
        # Middle finger tip (landmark 12)
        middle_tip = landmarks[12]
        middle_tip_pixel = (int(middle_tip.x * frame_width), int(middle_tip.y * frame_height))
        
        pixel_distance = self.calculate_distance(wrist_pixel, middle_tip_pixel)
        return self.pixel_to_cm(pixel_distance)
    
    def measure_finger_length(self, landmarks: List, frame_width: int, frame_height: int) -> float:
        """Measure index finger length from base to tip"""
        # Index finger base (landmark 5)
        index_base = landmarks[5]
        index_base_pixel = (int(index_base.x * frame_width), int(index_base.y * frame_height))
        
        # Index finger tip (landmark 8)
        index_tip = landmarks[8]
        index_tip_pixel = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
        
        pixel_distance = self.calculate_distance(index_base_pixel, index_tip_pixel)
        return self.pixel_to_cm(pixel_distance)
    
    def get_measurements(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """Get hand measurements from frame with dynamic calibration"""
        landmarks = self.get_hand_landmarks(frame)
        
        if landmarks is None:
            return None
        
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate palm width for dynamic calibration
        palm_width_pixels = self.get_palm_width_pixels(landmarks, frame_width, frame_height)
        
        # Update calibration based on palm width
        self.update_dynamic_calibration(palm_width_pixels)
        
        # Measure hand dimensions using current calibration
        hand_length = self.measure_hand_length(landmarks, frame_width, frame_height)
        finger_length = self.measure_finger_length(landmarks, frame_width, frame_height)
        
        measurement = {
            'hand_length': hand_length,
            'finger_length': finger_length,
            'palm_width_cm': self.pixel_to_cm(palm_width_pixels),
            'calibration_factor': self.calibration_factor,
            'calibration_confidence': self.get_calibration_confidence(),
            'timestamp': time.time()
        }
        
        # Add to history for stability
        self.measurement_history.append(measurement)
        if len(self.measurement_history) > self.max_history:
            self.measurement_history.pop(0)
        
        return self.get_averaged_measurements()
    
    def get_averaged_measurements(self) -> Dict[str, float]:
        """Get averaged measurements from recent history"""
        if not self.measurement_history:
            return None
        
        avg_hand_length = sum(m['hand_length'] for m in self.measurement_history) / len(self.measurement_history)
        avg_finger_length = sum(m['finger_length'] for m in self.measurement_history) / len(self.measurement_history)
        
        return {
            'hand_length': round(avg_hand_length, 1),
            'finger_length': round(avg_finger_length, 1),
            'confidence': min(1.0, len(self.measurement_history) / 3.0),
            'sample_count': len(self.measurement_history)
        }
    
    def draw_measurements(self, frame: np.ndarray, measurements: Dict[str, float]) -> np.ndarray:
        """Draw measurement overlays on frame"""
        if measurements is None:
            return frame
        
        # Draw measurement text
        y_offset = 30
        cv2.putText(frame, f"Hand Length: {measurements['hand_length']:.1f} cm", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Finger Length: {measurements['finger_length']:.1f} cm", 
                   (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Palm Width: {measurements.get('palm_width_cm', 0):.1f} cm", 
                   (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Calibration: {measurements.get('calibration_confidence', 0):.1%}", 
                   (10, y_offset + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Sample Count: {measurements['sample_count']}", 
                   (10, y_offset + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw hand landmarks
        landmarks = self.get_hand_landmarks(frame)
        if landmarks:
            frame_height, frame_width = frame.shape[:2]
            
            # Draw all landmarks
            for landmark in landmarks:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
            
            # Highlight measurement points
            # Wrist (landmark 0)
            wrist = landmarks[0]
            wrist_pixel = (int(wrist.x * frame_width), int(wrist.y * frame_height))
            cv2.circle(frame, wrist_pixel, 8, (0, 0, 255), -1)
            
            # Middle finger tip (landmark 12)
            middle_tip = landmarks[12]
            middle_tip_pixel = (int(middle_tip.x * frame_width), int(middle_tip.y * frame_height))
            cv2.circle(frame, middle_tip_pixel, 8, (0, 0, 255), -1)
            
            # Index finger base (landmark 5)
            index_base = landmarks[5]
            index_base_pixel = (int(index_base.x * frame_width), int(index_base.y * frame_height))
            cv2.circle(frame, index_base_pixel, 6, (255, 0, 255), -1)
            
            # Index finger tip (landmark 8)
            index_tip = landmarks[8]
            index_tip_pixel = (int(index_tip.x * frame_width), int(index_tip.y * frame_height))
            cv2.circle(frame, index_tip_pixel, 6, (255, 0, 255), -1)
            
            # Palm width measurement points
            pinky_base = landmarks[17]
            pinky_base_pixel = (int(pinky_base.x * frame_width), int(pinky_base.y * frame_height))
            cv2.circle(frame, pinky_base_pixel, 6, (255, 255, 0), -1)
            
            # Draw measurement lines
            cv2.line(frame, wrist_pixel, middle_tip_pixel, (0, 255, 0), 2)  # Hand length
            cv2.line(frame, index_base_pixel, index_tip_pixel, (255, 0, 255), 2)  # Finger length
            cv2.line(frame, index_base_pixel, pinky_base_pixel, (255, 255, 0), 2)  # Palm width
        
        return frame
    
    def run_measurement_session(self, duration_seconds: int = 10) -> Optional[Dict[str, float]]:
        """Run a measurement session"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return None
        
        print(f"🎯 Starting {duration_seconds}-second measurement session...")
        print("📋 Instructions:")
        print("  • Position your hand flat in front of the camera")
        print("  • Keep your hand visible and steady")
        print("  • The system will auto-calibrate using palm width")
        print("  • Press 'q' to quit early")
        
        start_time = time.time()
        measurements = None
        
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Get measurements
            current_measurements = self.get_hand_measurements(frame)
            if current_measurements:
                measurements = current_measurements
            
            # Draw overlays
            frame = self.draw_measurements(frame, measurements)
            
            # Show frame
            cv2.imshow('Golf Grip Measurement Tool', frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return measurements
    
    
    def get_hand_measurements(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """Get hand measurements from frame (alias for get_measurements)"""
        return self.get_measurements(frame)

def main():
    """Main function for testing"""
    print("🏌️ Golf Grip Measurement Tool with Dynamic Calibration")
    print("=" * 60)
    
    # Initialize measurement tool
    measurer = GolfGripMeasurement()
    
    print(f"📏 Using dynamic calibration based on palm width ({measurer.average_palm_width_cm}cm)")
    print("💡 The system will auto-calibrate for each hand position")
    
    # Run measurement session
    measurements = measurer.run_measurement_session(duration_seconds=10)
    
    if measurements:
        print("\n" + "=" * 60)
        print("📊 MEASUREMENT RESULTS")
        print("=" * 60)
        print(f"Hand Length: {measurements['hand_length']:.1f} cm")
        print(f"Finger Length: {measurements['finger_length']:.1f} cm")
        print(f"Palm Width: {measurements.get('palm_width_cm', 0):.1f} cm")
        print(f"Calibration Confidence: {measurements.get('calibration_confidence', 0):.1%}")
        print(f"Sample Count: {measurements['sample_count']}")
        
        # Validate measurements
        if measurements['confidence'] > 0.5:
            print("\n✅ Measurements look good!")
        else:
            print("\n⚠️  Low confidence - try better hand positioning")
        
        # Show grip engine format
        print("\n🔧 For Grip Recommendation Engine:")
        print(f"handLength: {measurements['hand_length']}")
        print(f"fingerLength: {measurements['finger_length']}")
        
        # Save measurements
        save_choice = input("\n💾 Save measurements to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"grip_measurements_{timestamp}.json"
            
            try:
                with open(filename, 'w') as f:
                    json.dump(measurements, f, indent=2)
                print(f"✅ Saved to: {filename}")
            except Exception as e:
                print(f"❌ Save failed: {e}")
        
    else:
        print("\n❌ No measurements collected")
        print("Possible issues:")
        print("  • Hand not visible in camera")
        print("  • Poor lighting conditions")
        print("  • Camera not working properly")
        print("  • Hand too close or too far from camera")

if __name__ == "__main__":
    main()
