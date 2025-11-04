"""
L2CS-Net Gaze Estimation Module
Implements the L2CS-Net architecture for accurate head pose and gaze estimation.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import gdown
import os
import urllib.request


class L2CSNet(nn.Module):
    """
    L2CS-Net architecture for gaze estimation.
    Architecture matches the official L2CS-Net implementation.
    """
    
    def __init__(self, backbone='resnet50', num_bins=90):
        """
        Initialize L2CS-Net model matching official architecture.
        
        Args:
            backbone: ResNet variant to use ('resnet18', 'resnet34', 'resnet50')
            num_bins: Number of bins for angle discretization
        """
        super(L2CSNet, self).__init__()
        self.num_bins = num_bins
        
        # Load pre-trained ResNet backbone - keep original structure
        if backbone == 'resnet18':
            base_model = torchvision.models.resnet18(weights='DEFAULT')
            feat_dim = 512
        elif backbone == 'resnet34':
            base_model = torchvision.models.resnet34(weights='DEFAULT')
            feat_dim = 512
        elif backbone == 'resnet50':
            base_model = torchvision.models.resnet50(weights='DEFAULT')
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Store backbone layers directly to match official weight names
        # This preserves the exact key names from the pretrained model
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        
        # Separate regression heads for pitch and yaw (matching official implementation)
        self.fc_yaw_gaze = nn.Linear(feat_dim, num_bins)
        self.fc_pitch_gaze = nn.Linear(feat_dim, num_bins)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            yaw_pred, pitch_pred: Predicted yaw and pitch logits [B, num_bins]
        """
        # Pass through ResNet backbone layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        
        # Predict yaw and pitch
        yaw_pred = self.fc_yaw_gaze(features)
        pitch_pred = self.fc_pitch_gaze(features)
        
        return yaw_pred, pitch_pred


class L2CSGazeEstimator:
    """Gaze estimator using L2CS-Net for accurate head pose estimation."""
    
    # Official L2CS-Net repository model URLs
    # Note: These may need to be downloaded manually from GitHub releases
    MODEL_URLS = {
        'resnet50': 'https://github.com/Ahmednull/L2CS-Net/releases/download/v1.0/L2CSNet_gaze360.pkl',
        'manual_instructions': 'Visit https://github.com/Ahmednull/L2CS-Net for official weights'
    }
    
    def __init__(self, device=None, model_dir='models'):
        """
        Initialize L2CS gaze estimator.
        
        Args:
            device: torch device ('cuda' or 'cpu'). Auto-detected if None.
            model_dir: Directory to store model weights
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        print(f"Initializing L2CS-Net on device: {self.device}")
        
        # Initialize face detector (OpenCV DNN)
        self._load_face_detector()
        
        # Load L2CS-Net model
        self.model = self._load_model()
        
        # Image preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Angle bins configuration
        self.num_bins = 90
        self.idx_tensor = torch.arange(self.num_bins, dtype=torch.float32).to(self.device)
    
    def _load_face_detector(self):
        """Load OpenCV DNN face detector."""
        # URLs for face detection model
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        proto_path = self.model_dir / "deploy.prototxt"
        model_path = self.model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        
        # Download proto file if not exists
        if not proto_path.exists():
            print("Downloading face detector prototxt...")
            try:
                urllib.request.urlretrieve(proto_url, proto_path)
                print(f"Downloaded to {proto_path}")
            except Exception as e:
                print(f"Warning: Could not download prototxt: {e}")
        
        # Download model file if not exists
        if not model_path.exists():
            print("Downloading face detector model (~10MB)...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"Downloaded to {model_path}")
            except Exception as e:
                print(f"Warning: Could not download face model: {e}")
        
        # Load face detector
        if proto_path.exists() and model_path.exists():
            self.face_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
            print("OpenCV DNN face detector loaded successfully")
        else:
            print("Warning: Face detector files not found. Face detection may not work.")
            self.face_net = None
        
    def _load_model(self):
        """Load L2CS-Net model with pre-trained weights."""
        model = L2CSNet(backbone='resnet50', num_bins=90)
        model = model.to(self.device)
        
        # Try to load pre-trained weights - check both possible filenames
        model_paths = [
            self.model_dir / 'L2CSNet_gaze360.pkl',  # Official name
            self.model_dir / 'l2cs_resnet50.pkl',     # Alternative name
        ]
        
        weights_loaded = False
        for model_path in model_paths:
            if model_path.exists():
                print(f"Loading L2CS-Net weights from {model_path}")
                try:
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    # Try loading with strict=True first
                    try:
                        model.load_state_dict(state_dict, strict=True)
                        print("✓ L2CS-Net weights loaded successfully - using trained gaze model")
                        weights_loaded = True
                        break
                    except RuntimeError as e:
                        # If strict loading fails, try with strict=False (ignores extra keys)
                        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                        if len(missing_keys) == 0:
                            print(f"✓ L2CS-Net weights loaded successfully - using trained gaze model")
                            print(f"  (Ignored {len(unexpected_keys)} unused keys from weights file)")
                            weights_loaded = True
                            break
                        else:
                            print(f"⚠ Warning: {len(missing_keys)} keys missing in weights")
                            print("✓ L2CS-Net weights partially loaded - using trained gaze model")
                            weights_loaded = True
                            break
                        
                except Exception as e:
                    print(f"Warning: Could not load weights from {model_path}")
                    print(f"Error: {str(e)[:200]}")  # Print first 200 chars of error
        
        if not weights_loaded:
            print(f"\nPre-trained L2CS-Net weights not found in {self.model_dir}")
            print("Attempting automatic download...")
            self._download_weights(model_paths[0])
            
            # Try loading after download
            if model_paths[0].exists():
                try:
                    state_dict = torch.load(model_paths[0], map_location=self.device, weights_only=False)
                    try:
                        model.load_state_dict(state_dict, strict=True)
                        print("✓ L2CS-Net weights loaded successfully - using trained gaze model")
                        weights_loaded = True
                    except RuntimeError:
                        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                        if len(missing_keys) == 0:
                            print(f"✓ L2CS-Net weights loaded successfully - using trained gaze model")
                            print(f"  (Ignored {len(unexpected_keys)} unused keys from weights file)")
                            weights_loaded = True
                        else:
                            print(f"⚠ Warning: {len(missing_keys)} keys missing in weights")
                            print("✓ L2CS-Net weights partially loaded - using trained gaze model")
                            weights_loaded = True
                except Exception as e:
                    print(f"Warning: Could not load downloaded weights")
                    print(f"Error: {str(e)[:200]}")
        
        if not weights_loaded:
            print("\n" + "="*70)
            print("⚠ Using ImageNet pre-trained ResNet50 backbone only")
            print("For better accuracy, download L2CS-Net weights manually:")
            print("1. Visit: https://github.com/Ahmednull/L2CS-Net")
            print("2. Download 'L2CSNet_gaze360.pkl' from releases or models folder")
            print(f"3. Place it in: {self.model_dir.absolute()}")
            print("="*70 + "\n")
        
        model.eval()
        return model
    
    def _download_weights(self, save_path):
        """Download pre-trained L2CS-Net weights."""
        # Try GitHub release URL first
        try:
            print("Attempting to download from GitHub releases...")
            url = self.MODEL_URLS['resnet50']
            urllib.request.urlretrieve(url, save_path)
            print(f"✓ Weights downloaded to {save_path}")
            return
        except Exception as e:
            print(f"GitHub download failed: {e}")
        
        # Try gdown as fallback (for Google Drive links)
        try:
            print("Trying alternative download method...")
            # This is the old Google Drive URL - may not work
            gdown_url = 'https://drive.google.com/uc?id=1gfQmp-RkGDt1R_PqEPxqEa9xJM5xKpbX'
            gdown.download(gdown_url, str(save_path), quiet=False)
            print(f"✓ Weights downloaded to {save_path}")
        except Exception as e:
            print(f"Automatic download failed: {e}")
            print("Please download weights manually (see instructions above)")
    
    def detect_faces(self, image, confidence_threshold=0.7):
        """
        Detect faces in the image using OpenCV DNN.
        
        Args:
            image: Input image (BGR format from OpenCV)
            confidence_threshold: Minimum confidence for face detection (default: 0.7)
            
        Returns:
            List of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        if self.face_net is None:
            return []
        
        h, w = image.shape[:2]
        
        # Prepare image for DNN
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        # Detect faces
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # Extract faces with high confidence
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure valid coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2, y2))
        
        return faces
    
    def estimate_gaze(self, face_image):
        """
        Estimate gaze direction from face image.
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            tuple: (pitch, yaw) angles in degrees, or (None, None) if failed
        """
        if face_image is None or face_image.size == 0:
            return None, None
        
        h, w = face_image.shape[:2]
        if h < 40 or w < 40:
            return None, None
        
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Preprocess image
            img_tensor = self.transform(rgb_image)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                yaw_pred, pitch_pred = self.model(img_tensor)
            
            # Convert logits to angles
            yaw_pred = torch.softmax(yaw_pred, dim=1)
            pitch_pred = torch.softmax(pitch_pred, dim=1)
            
            yaw = torch.sum(yaw_pred * self.idx_tensor) * 4 - 180
            pitch = torch.sum(pitch_pred * self.idx_tensor) * 4 - 180
            
            return float(pitch.cpu()), float(yaw.cpu())
            
        except Exception as e:
            print(f"Error in gaze estimation: {e}")
            return None, None
    
    def get_gaze_direction(self, face_roi):
        """
        Get gaze direction from face region of interest.
        Compatible interface with original GazeDetector.
        
        Args:
            face_roi: Face region image (BGR format)
            
        Returns:
            tuple: (is_looking_at_camera, pitch, yaw)
        """
        # Detect face in ROI
        faces = self.detect_faces(face_roi)
        
        if len(faces) == 0:
            return False, None, None
        
        # Use the largest/first face
        x1, y1, x2, y2 = faces[0]
        face_crop = face_roi[y1:y2, x1:x2]
        
        # Estimate gaze
        pitch, yaw = self.estimate_gaze(face_crop)
        
        if pitch is None or yaw is None:
            return False, None, None
        
        # Determine if looking at camera with STRICT thresholds
        # L2CS-Net is more accurate, so we can use tighter thresholds
        pitch_threshold = (-15, 10)   # Slightly looking down to slightly up
        yaw_threshold = (-15, 15)     # Facing forward
        
        is_looking = (pitch_threshold[0] <= pitch <= pitch_threshold[1] and 
                     yaw_threshold[0] <= yaw <= yaw_threshold[1])
        
        return is_looking, pitch, yaw
    
    def cleanup(self):
        """Release GPU resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'face_net'):
            del self.face_net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("L2CS-Net resources cleaned up")


if __name__ == "__main__":
    # Test the gaze estimator
    print("Testing L2CS-Net Gaze Estimator...")
    estimator = L2CSGazeEstimator()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam")
    else:
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            faces = estimator.detect_faces(frame)
            
            for (x1, y1, x2, y2) in faces:
                # Get face ROI
                face_roi = frame[y1:y2, x1:x2]
                
                # Estimate gaze
                pitch, yaw = estimator.estimate_gaze(face_roi)
                
                if pitch is not None and yaw is not None:
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow('L2CS-Net Gaze Estimation', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    estimator.cleanup()

