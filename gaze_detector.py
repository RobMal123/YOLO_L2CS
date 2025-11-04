"""
YOLO Gaze Detection System
Detects humans in video and marks those looking at the camera.
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import urllib.request
import os
import math

# Import L2CS gaze estimator
try:
    from l2cs_gaze import L2CSGazeEstimator
    L2CS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: L2CS-Net not available: {e}")
    print("Falling back to Haar Cascades method")
    L2CS_AVAILABLE = False


class GazeDetector:
    """Detects human gaze direction using OpenCV face and eye detection."""
    
    def __init__(self):
        """Initialize OpenCV face and eye detectors."""
        # Download cascade files if not present
        self._ensure_cascades()
        
        # Load Haar Cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
    def _ensure_cascades(self):
        """Ensure cascade files are available."""
        # OpenCV includes these by default, no action needed
        pass
        
    def get_gaze_direction(self, face_roi):
        """
        Estimate gaze direction from face region.
        
        Args:
            face_roi: Cropped face image (BGR format)
            
        Returns:
            tuple: (is_looking_at_camera, pitch, yaw, face_bbox) or (False, None, None, None)
        """
        if face_roi is None or face_roi.size == 0:
            return False, None, None, None
            
        h, w = face_roi.shape[:2]
        
        # Minimum size check
        if h < 40 or w < 40:
            return False, None, None, None
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect face within the ROI with stricter parameters
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,  # More strict: increased from 3 to 5
            minSize=(40, 40)  # Larger minimum size for better detection
        )
        
        if len(faces) == 0:
            # No face detected in ROI - person likely not facing camera
            return False, None, None, None
        
        # Use the largest face detected
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        fx, fy, fw, fh = largest_face
        face_bbox = (fx, fy, fx + fw, fy + fh)  # Convert to (x1, y1, x2, y2) format
        
        # Extract face region for eye detection
        face_gray = gray[fy:fy+fh, fx:fx+fw]
        
        # Detect eyes with stricter parameters
        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.05,  # Smaller scale for more precision
            minNeighbors=5,    # More strict: increased from 3 to 5
            minSize=(20, 20)   # Larger minimum size for better detection
        )
        
        # Require at least TWO eyes to be detected for looking at camera
        # This ensures frontal view
        if len(eyes) < 2:
            # Less than 2 eyes visible - likely not facing camera
            return False, None, None, face_bbox
        
        # Check face aspect ratio - frontal faces should be wider
        face_aspect_ratio = fw / fh if fh > 0 else 0
        if face_aspect_ratio < 0.65:
            # Face too narrow - likely side profile
            return False, None, None, face_bbox
        
        # Calculate head pose based on face and eye positions
        pitch, yaw = self._calculate_head_pose(fx, fy, fw, fh, eyes, w, h)
        
        # Determine if looking at camera with STRICT thresholds
        pitch_threshold = (-20, 5)   # More restrictive: slightly below to neutral
        yaw_threshold = (-20, 20)    # More restrictive: reduced side tolerance
        
        is_looking = (pitch_threshold[0] <= pitch <= pitch_threshold[1] and 
                     yaw_threshold[0] <= yaw <= yaw_threshold[1])
        
        return is_looking, pitch, yaw, face_bbox
    
    def _calculate_head_pose(self, fx, fy, fw, fh, eyes, roi_w, roi_h):
        """
        Calculate approximate head pose angles.
        
        Args:
            fx, fy, fw, fh: Face bounding box
            eyes: Detected eyes
            roi_w, roi_h: ROI dimensions
            
        Returns:
            tuple: (pitch, yaw) in degrees
        """
        # Calculate face center
        face_center_x = fx + fw / 2
        face_center_y = fy + fh / 2
        
        # Calculate face aspect ratio (width/height)
        # Frontal faces: ratio ~0.75-0.85, Side profiles: ratio <0.6
        face_aspect_ratio = fw / fh if fh > 0 else 1.0
        
        # Calculate horizontal offset from center (yaw)
        roi_center_x = roi_w / 2
        horizontal_offset = (face_center_x - roi_center_x) / roi_w
        yaw = horizontal_offset * 60  # Scale to approximate degrees
        
        # Adjust yaw based on face aspect ratio
        # Narrower face = larger yaw angle (turned away)
        if face_aspect_ratio < 0.7:
            # Face is narrow, likely turned to the side
            yaw_penalty = (0.7 - face_aspect_ratio) * 100
            if horizontal_offset < 0:
                yaw -= yaw_penalty
            else:
                yaw += yaw_penalty
        
        # Calculate vertical offset from center (pitch)
        roi_center_y = roi_h / 2
        vertical_offset = (face_center_y - roi_center_y) / roi_h
        pitch = vertical_offset * 45  # Scale to approximate degrees
        
        # Adjust based on eye detection
        if len(eyes) >= 2:
            # Get two most prominent eyes
            sorted_eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
            eye1, eye2 = sorted_eyes[0], sorted_eyes[1]
            
            # Calculate eye positions
            eye1_x, eye1_y = eye1[0] + eye1[2]/2, eye1[1] + eye1[3]/2
            eye2_x, eye2_y = eye2[0] + eye2[2]/2, eye2[1] + eye2[3]/2
            
            # Calculate horizontal distance between eyes
            eye_distance = abs(eye1_x - eye2_x)
            expected_eye_distance = fw * 0.4  # Eyes should be ~40% of face width apart
            
            # If eyes are too close together, person might be facing away
            if eye_distance < expected_eye_distance * 0.6:
                yaw_adjustment = 40  # Increased penalty from 30 to 40
                if face_center_x < roi_center_x:
                    yaw -= yaw_adjustment
                else:
                    yaw += yaw_adjustment
            
            # Check vertical alignment of eyes
            eye_vertical_diff = abs(eye1_y - eye2_y)
            # If eyes are not horizontally aligned (vertical diff too large), head is tilted
            if eye_vertical_diff > fh * 0.2:  # More than 20% of face height
                # Apply pitch adjustment for tilted head
                pitch += 15
            
            # Adjust pitch based on eye vertical position
            avg_eye_y = (eye1_y + eye2_y) / 2
            eye_ratio = avg_eye_y / fh
            
            # Eyes in upper portion (< 0.4) suggests looking down/neutral
            # Eyes in lower portion (> 0.6) suggests looking up
            if eye_ratio < 0.4:
                pitch -= 10
            elif eye_ratio > 0.6:
                pitch += 10
        
        return pitch, yaw
    
    def cleanup(self):
        """Release resources."""
        # No resources to release for Haar Cascades
        pass


class HybridGazeDetector:
    """
    Hybrid gaze detector combining L2CS-Net with eye detection validation.
    Uses L2CS-Net for accurate head pose angles and eye detection for validation.
    """
    
    def __init__(self, use_l2cs=True, debug=False, require_two_eyes=False):
        """
        Initialize hybrid gaze detector.
        
        Args:
            use_l2cs: If True, use L2CS-Net. If False, use legacy Haar Cascades.
            debug: If True, print debug information during detection.
            require_two_eyes: If True, require both eyes. If False, more lenient (recommended for L2CS).
        """
        self.use_l2cs = use_l2cs and L2CS_AVAILABLE
        self.debug = debug
        self.require_two_eyes = require_two_eyes
        self.stats = {'faces_detected': 0, 'eyes_detected': 0, 'angle_passed': 0, 'final_passed': 0}
        
        if self.use_l2cs:
            print("Initializing Hybrid Gaze Detector (L2CS-Net + Eye Validation)")
            self.l2cs_estimator = L2CSGazeEstimator()
            # Also initialize eye detector for validation
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        else:
            print("Initializing Legacy Gaze Detector (Haar Cascades)")
            self.legacy_detector = GazeDetector()
    
    def print_stats(self):
        """Print detection statistics."""
        if self.use_l2cs and self.stats['faces_detected'] > 0:
            print("\nL2CS Detection Statistics:")
            print(f"  Faces detected: {self.stats['faces_detected']}")
            print(f"  Eyes detected (2+): {self.stats['eyes_detected']}")
            print(f"  Passed angle thresholds: {self.stats['angle_passed']}")
            print(f"  Final detections: {self.stats['final_passed']}")
    
    def get_gaze_direction(self, face_roi):
        """
        Get gaze direction using hybrid approach or legacy method.
        
        Args:
            face_roi: Cropped face image (BGR format)
            
        Returns:
            tuple: (is_looking_at_camera, pitch, yaw, face_bbox) 
                   where face_bbox is (x1, y1, x2, y2) or None
        """
        if not self.use_l2cs:
            # Use legacy Haar Cascades method
            return self.legacy_detector.get_gaze_direction(face_roi)
        
        # Hybrid approach: L2CS-Net + Eye validation
        if face_roi is None or face_roi.size == 0:
            return False, None, None, None
        
        h, w = face_roi.shape[:2]
        if h < 40 or w < 40:
            return False, None, None, None
        
        # Step 1: Get pitch and yaw from L2CS-Net
        faces = self.l2cs_estimator.detect_faces(face_roi)
        
        if len(faces) == 0:
            return False, None, None, None
        
        self.stats['faces_detected'] += 1
        
        # Use the largest face
        x1, y1, x2, y2 = faces[0]
        face_bbox = (x1, y1, x2, y2)  # Store for visualization
        face_crop = face_roi[y1:y2, x1:x2]
        
        pitch, yaw = self.l2cs_estimator.estimate_gaze(face_crop)
        
        if pitch is None or yaw is None:
            return False, None, None, face_bbox
        
        # Step 2: Validate with eye detection (ensures frontal view)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,  # Reduced from 4 to be more lenient
            minSize=(15, 15)  # Reduced from 20 to detect smaller eyes
        )
        
        # Check eye requirement based on setting
        min_eyes_required = 2 if self.require_two_eyes else 1
        if len(eyes) < min_eyes_required:
            # Not enough eyes visible - likely side profile or occlusion
            if self.debug:
                print(f"  Face detected but only {len(eyes)} eye(s) - skipping (need {min_eyes_required})")
            return False, pitch, yaw, face_bbox
        
        self.stats['eyes_detected'] += 1
        
        # Step 3: Apply thresholds (tighter because L2CS is more accurate)
        pitch_threshold = (-15, 10)   # Slightly looking down to slightly up
        yaw_threshold = (-15, 15)     # Facing forward
        
        is_looking = (pitch_threshold[0] <= pitch <= pitch_threshold[1] and 
                     yaw_threshold[0] <= yaw <= yaw_threshold[1])
        
        if is_looking:
            self.stats['angle_passed'] += 1
            self.stats['final_passed'] += 1
        elif self.debug:
            print(f"  Angles outside threshold: P={pitch:.1f} Y={yaw:.1f}")
        
        return is_looking, pitch, yaw, face_bbox
    
    def cleanup(self):
        """Release resources."""
        self.print_stats()
        if self.use_l2cs:
            self.l2cs_estimator.cleanup()
        else:
            self.legacy_detector.cleanup()


class VideoProcessor:
    """Processes video to detect humans and mark those looking at camera."""
    
    def __init__(self, yolo_model='yolov8n.pt', confidence=0.5, gaze_method='l2cs', require_two_eyes=False):
        """
        Initialize video processor.
        
        Args:
            yolo_model: Path to YOLO model or model name
            confidence: Detection confidence threshold
            gaze_method: Gaze estimation method ('l2cs' or 'legacy')
            require_two_eyes: If True, require both eyes visible (stricter)
        """
        print(f"Loading YOLO model: {yolo_model}")
        self.yolo = YOLO(yolo_model)
        self.confidence = confidence
        self.gaze_method = gaze_method
        
        # Initialize gaze detector
        use_l2cs = (gaze_method == 'l2cs')
        self.gaze_detector = HybridGazeDetector(use_l2cs=use_l2cs, require_two_eyes=require_two_eyes)
        
    def process_video(self, input_path, output_path):
        """
        Process video file and create output with marked individuals.
        
        Args:
            input_path: Path to input video file
            output_path: Path to save output video
        """
        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detected_looking_count = 0
        
        print("Processing video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Run YOLO detection (class 0 is 'person' in COCO dataset)
            results = self.yolo(frame, conf=self.confidence, classes=[0], verbose=False)
            
            # Process each detected person
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    
                    # Ensure valid coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Extract person region
                    person_roi = frame[y1:y2, x1:x2]
                    
                    # Focus on upper portion for face detection (top 40% of person)
                    face_height = int((y2 - y1) * 0.4)
                    face_roi = person_roi[:face_height, :]
                    
                    # Check gaze direction (also returns face bbox for visualization)
                    is_looking, pitch, yaw, face_bbox = self.gaze_detector.get_gaze_direction(face_roi)
                    
                    # Mark person if looking at camera
                    if is_looking:
                        detected_looking_count += 1
                        
                        # Draw green bounding box around person
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Draw face bounding box in yellow/cyan
                        if face_bbox is not None:
                            fx1, fy1, fx2, fy2 = face_bbox
                            # Convert face coords from ROI to frame coordinates
                            fx1_abs = x1 + fx1
                            fy1_abs = y1 + fy1
                            fx2_abs = x1 + fx2
                            fy2_abs = y1 + fy2
                            cv2.rectangle(frame, (fx1_abs, fy1_abs), (fx2_abs, fy2_abs), (0, 255, 255), 2)
                            
                            # Draw gaze direction arrow
                            if pitch is not None and yaw is not None:
                                # Calculate face center
                                face_center_x = (fx1_abs + fx2_abs) // 2
                                face_center_y = (fy1_abs + fy2_abs) // 2
                                
                                # Calculate arrow endpoint based on pitch and yaw
                                # Scale factor for arrow length
                                arrow_length = 80
                                
                                # Convert angles to arrow direction
                                # Yaw: left (-) to right (+)
                                # Pitch: down (-) to up (+)
                                yaw_rad = math.radians(yaw)
                                pitch_rad = math.radians(-pitch)  # Negative because image Y increases downward
                                
                                # Calculate arrow end point
                                end_x = int(face_center_x + arrow_length * math.sin(yaw_rad))
                                end_y = int(face_center_y + arrow_length * math.sin(pitch_rad))
                                
                                # Draw arrow
                                cv2.arrowedLine(frame, (face_center_x, face_center_y), 
                                              (end_x, end_y), (0, 255, 255), 2, tipLength=0.3)
                        
                        # Add label with method indicator
                        method_tag = "[L2CS]" if self.gaze_method == 'l2cs' else "[Legacy]"
                        label = f"{method_tag} Looking at camera (conf: {conf:.2f})"
                        if pitch is not None and yaw is not None:
                            label += f" P:{pitch:.1f} Y:{yaw:.1f}"
                        
                        # Draw label background
                        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Write processed frame
            out.write(frame)
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Cleanup
        cap.release()
        out.release()
        self.gaze_detector.cleanup()
        
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Instances of people looking at camera: {detected_looking_count}")
        print(f"Output saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect humans in video and mark those looking at the camera"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output video file"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8n.pt",
        help="YOLO model to use (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--gaze-method", "-g",
        choices=['l2cs', 'legacy'],
        default='l2cs',
        help="Gaze estimation method: 'l2cs' (L2CS-Net + eye validation) or 'legacy' (Haar Cascades). Default: l2cs"
    )
    parser.add_argument(
        "--require-two-eyes",
        action='store_true',
        help="Require both eyes visible (stricter but more false negatives). Default: False (only 1 eye needed - more detections)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Create output directory if needed
    output_dir = Path(args.output).parent
    if output_dir != Path('.') and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process video
    processor = VideoProcessor(
        yolo_model=args.model, 
        confidence=args.confidence,
        gaze_method=args.gaze_method,
        require_two_eyes=args.require_two_eyes
    )
    processor.process_video(args.input, args.output)


if __name__ == "__main__":
    main()

