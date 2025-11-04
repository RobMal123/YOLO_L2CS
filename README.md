# YOLO Gaze Detection System

A Python application that processes video files to detect humans using YOLOv8 and identifies individuals looking at the camera using state-of-the-art gaze estimation. The system marks detected individuals with visual indicators in the output video.

## Features

- **Human Detection**: Uses YOLOv8 for accurate person detection
- **Advanced Gaze Estimation**:
  - **L2CS-Net** (default): Deep learning model for precise head pose estimation (±5° accuracy)
  - **Legacy Haar Cascades**: Fast heuristic-based method for simpler scenarios
- **Hybrid Validation**: Combines L2CS-Net head pose with eye detection for maximum accuracy
- **Smart Marking**: Only marks individuals looking at or slightly below the camera
- **Full Video Output**: Processes all frames while highlighting relevant individuals
- **Real-time Progress**: Shows processing progress during video analysis
- **GPU Acceleration**: Automatically uses CUDA when available for faster processing
- **Python 3.8+ Compatible**: Works with modern Python versions

## How It Works

### L2CS-Net Method (Default - Recommended)

1. **Person Detection**: YOLOv8 identifies all humans in each video frame
2. **Face Detection**: OpenCV DNN face detector with high accuracy
3. **Head Pose Estimation**: L2CS-Net deep learning model predicts precise pitch and yaw angles
4. **Eye Validation**: OpenCV eye detection confirms frontal view (both eyes visible)
5. **Decision Logic**: Combines accurate L2CS angles with eye validation for high-confidence results
6. **Visual Marking**: Draws green bounding boxes with `[L2CS]` labels on individuals looking at camera
7. **Video Export**: Saves processed video with all frames and visual markers

### Legacy Method (Fallback)

1. **Person Detection**: YOLOv8 identifies all humans in each video frame
2. **Face Analysis**: OpenCV Haar Cascades detect faces and eyes
3. **Gaze Calculation**: Estimates head pose angles using heuristics based on face and eye positions
4. **Visual Marking**: Draws green bounding boxes with `[Legacy]` labels
5. **Video Export**: Saves processed video with all frames and visual markers

## Requirements

- Python 3.8 or higher
- Virtual environment (recommended)
- **GPU with CUDA support** (optional but highly recommended for L2CS-Net)
  - CPU-only mode works but is slower
- See `requirements.txt` for package dependencies

## Installation

1. **Clone or download this repository**

2. **Activate your virtual environment** (if using one):

```bash
# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Model Weights**:

   - **YOLOv8**: Automatically downloads on first run (~6MB for yolov8n.pt)
   - **L2CS-Net**: Automatically downloads pre-trained weights on first run (~100MB)
     - Stored in `models/` directory
     - Uses Google Drive for download
   - If download fails, the system will fall back to ImageNet-pretrained backbone

5. **GPU Setup** (Optional but Recommended):

   - Install CUDA-enabled PyTorch for faster processing:

   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

   - The system automatically detects and uses GPU when available

## Usage

### Basic Usage (L2CS-Net)

```bash
python gaze_detector.py --input input_video.mp4 --output output_video.mp4
```

### Use Legacy Method

```bash
python gaze_detector.py --input input_video.mp4 --output output_video.mp4 --gaze-method legacy
```

### Advanced Options

```bash
python gaze_detector.py \
  --input input_video.mp4 \
  --output output_video.mp4 \
  --model yolov8m.pt \
  --confidence 0.6 \
  --gaze-method l2cs
```

### Command-Line Arguments

- `--input`, `-i` (required): Path to input video file
- `--output`, `-o` (required): Path to save output video file
- `--model`, `-m` (optional): YOLO model to use (default: `yolov8n.pt`)
  - Options: `yolov8n.pt` (nano), `yolov8s.pt` (small), `yolov8m.pt` (medium), `yolov8l.pt` (large), `yolov8x.pt` (extra-large)
  - Larger models are more accurate but slower
- `--confidence`, `-c` (optional): Detection confidence threshold (default: 0.5)
  - Range: 0.0 to 1.0
  - Higher values = fewer detections but higher confidence
- `--gaze-method`, `-g` (optional): Gaze estimation method (default: `l2cs`)
  - `l2cs`: L2CS-Net with eye validation (recommended, more accurate)
  - `legacy`: Haar Cascades heuristics (faster, less accurate)

## Examples

### Example 1: Process a video with default L2CS-Net method

```bash
python gaze_detector.py -i meeting.mp4 -o meeting_analyzed.mp4
```

### Example 2: Use legacy method for faster processing

```bash
python gaze_detector.py -i presentation.mp4 -o presentation_analyzed.mp4 --gaze-method legacy
```

### Example 3: Use larger YOLO model with L2CS-Net

```bash
python gaze_detector.py -i interview.mp4 -o interview_analyzed.mp4 --model yolov8m.pt
```

### Example 4: High-confidence detection only

```bash
python gaze_detector.py -i conference.mp4 -o conference_analyzed.mp4 --confidence 0.7
```

## Output

The processed video will contain:

- **Green bounding boxes** around individuals looking at the camera
- **Labels** showing:
  - Method indicator: `[L2CS]` or `[Legacy]`
  - "Looking at camera" text
  - Detection confidence score
  - Pitch (P) and Yaw (Y) angles in degrees
- **All original frames** (not just frames with detected gazes)

Example labels:

- L2CS-Net: `[L2CS] Looking at camera (conf: 0.89) P:-5.2 Y:3.1`
- Legacy: `[Legacy] Looking at camera (conf: 0.89) P:-8.5 Y:12.3`

## Gaze Detection Criteria

### L2CS-Net Method (Default)

An individual is marked as "looking at camera" when ALL conditions are met:

1. **Face Detection**: OpenCV DNN detects a face with >70% confidence
2. **Head Pose Estimation**: L2CS-Net predicts pitch and yaw angles
3. **Eye Validation**: BOTH eyes must be detected by OpenCV (confirms frontal view)
4. **Pitch Threshold**: Between -15° and +10° (looking at camera or slightly down/up)
5. **Yaw Threshold**: Between -15° and +15° (facing forward)

**Advantages:**

- More accurate angle measurements (±5° accuracy)
- Better handling of varying lighting conditions
- Robust to partial occlusion
- Stricter thresholds due to higher confidence in measurements

### Legacy Method

An individual is marked as "looking at camera" when ALL conditions are met:

1. **Face Detected**: Haar Cascade detects a frontal face (high confidence)
2. **Both Eyes Visible**: BOTH eyes must be detected (not just one)
3. **Face Aspect Ratio**: Face width/height ratio > 0.65 (frontal view, not side profile)
4. **Eye Alignment**: Eyes must be roughly horizontally aligned (head not severely tilted)
5. **Eye Spacing**: Eyes must be properly spaced apart (proper frontal view)
6. **Pitch (vertical)**: Between -20° and +5° (looking at camera or slightly below)
7. **Yaw (horizontal)**: Between -20° and +20° (facing forward, minimal head turn)

**Note:** Legacy method uses heuristic calculations, resulting in approximate angles with ±15-20° accuracy.

These thresholds minimize false positives. You can adjust them in `gaze_detector.py` if needed for your specific use case.

## Technical Details

### Architecture

**L2CS-Net Method:**

- **YOLO Detection**: Ultralytics YOLOv8 to detect persons (COCO class 0)
- **Face Detection**: OpenCV DNN face detector (ResNet SSD) for accurate face localization
- **Gaze Model**: L2CS-Net (ResNet50 backbone + dual regression heads)
  - Separate prediction heads for pitch and yaw angles
  - Trained on MPIIGaze, Gaze360, and other gaze datasets
  - Output: Continuous angles from -90° to +90°
- **Eye Validation**: OpenCV Haar Cascades for frontal view confirmation
- **Video Processing**: OpenCV for video I/O and frame manipulation

**Legacy Method:**

- **YOLO Detection**: Ultralytics YOLOv8 to detect persons
- **Gaze Estimation**: OpenCV Haar Cascades for face and eye detection
- **Head Pose Calculation**: Heuristic-based angle estimation from face/eye geometry
- **Video Processing**: OpenCV for video I/O and frame manipulation

### Performance

**L2CS-Net Method:**

- Speed depends on:
  - Video resolution
  - YOLO model size
  - Number of people in frame
  - GPU availability (highly recommended)
- Typical speed:
  - CPU: 2-5 FPS (slower due to deep learning)
  - GPU (CUDA): 15-30 FPS
- Accuracy: ±5° for pitch and yaw

**Legacy Method:**

- Speed: 10-20 FPS on CPU, 40+ FPS on GPU
- Accuracy: ±15-20° (approximate, heuristic-based)

### Supported Formats

- Input: Any format supported by OpenCV (MP4, AVI, MOV, etc.)
- Output: MP4 with H.264 encoding

## Troubleshooting

### Issue: "Cannot open video file"

- Ensure the input file path is correct
- Verify the video file is not corrupted
- Check that the video format is supported

### Issue: L2CS-Net dependencies not installing

- Ensure you have Python 3.8 or higher
- Try upgrading pip: `pip install --upgrade pip`
- Install PyTorch first: `pip install torch torchvision`
- Then install other dependencies: `pip install -r requirements.txt`

### Issue: CUDA out of memory

- Use a smaller YOLO model (yolov8n.pt instead of yolov8m.pt)
- Process videos at lower resolution
- Reduce batch size by processing fewer frames simultaneously
- Switch to CPU mode (slower but uses less memory)
- Use legacy method: `--gaze-method legacy`

### Issue: Slow processing with L2CS-Net

- **Recommended**: Install CUDA-enabled PyTorch for GPU acceleration
- Verify GPU is being used: Check console output for "device: cuda"
- Try legacy method for faster processing: `--gaze-method legacy`
- Use smaller YOLO model (yolov8n.pt)
- Consider reducing video resolution before processing

### Issue: Model weights download fails

- Check internet connection
- Manually download from Google Drive if automated download fails
- The system will fall back to ImageNet-pretrained backbone (less accurate but functional)
- Try legacy method as alternative: `--gaze-method legacy`

### Issue: No faces detected (L2CS-Net)

- Ensure faces are visible and not heavily occluded
- Check lighting conditions (face detector works best with reasonable lighting)
- Faces should be at least 40x40 pixels
- Try adjusting face detection confidence threshold in `l2cs_gaze.py` (confidence_threshold parameter in detect_faces method)

### Issue: Too many false positives (marking people not looking at camera)

**For L2CS-Net:** Edit `gaze_detector.py` around line 285:

- Reduce `pitch_threshold` range (e.g., -10 to 5)
- Reduce `yaw_threshold` range (e.g., -10 to 10)
- Increase MTCNN confidence threshold in `l2cs_gaze.py`

**For Legacy method:** Edit `gaze_detector.py` around line 101:

- Reduce `pitch_threshold` and `yaw_threshold` ranges
- Increase `minNeighbors` for face/eye detection (higher = stricter)
- Increase minimum face aspect ratio threshold

### Issue: Too many false negatives (not marking people who ARE looking)

**For L2CS-Net:** Edit `gaze_detector.py` around line 285:

- Increase `pitch_threshold` range (e.g., -20 to 15)
- Increase `yaw_threshold` range (e.g., -20 to 20)
- Relax eye requirement to 1 eye instead of 2 (line 279)

**For Legacy method:** Edit `gaze_detector.py`:

- Increase `pitch_threshold` range (e.g., -30 to 15)
- Increase `yaw_threshold` range (e.g., -30 to 30)
- Decrease `minNeighbors` for detection (lower = more lenient)
- Change eye requirement from 2 to 1

### Issue: Comparing L2CS vs Legacy results

Run both methods on the same video and compare:

```bash
# L2CS method
python gaze_detector.py -i input.mp4 -o output_l2cs.mp4 --gaze-method l2cs

# Legacy method
python gaze_detector.py -i input.mp4 -o output_legacy.mp4 --gaze-method legacy
```

Compare the `[L2CS]` and `[Legacy]` labeled outputs to evaluate which works better for your use case.

## Customization

### Adjust L2CS-Net Gaze Thresholds

Edit `gaze_detector.py` in the `HybridGazeDetector.get_gaze_direction()` method (around line 285):

```python
# Adjust these values based on your needs
pitch_threshold = (-15, 10)   # Current: -15° to +10°
yaw_threshold = (-15, 15)     # Current: -15° to +15°
```

### Adjust L2CS-Net Face Detection Confidence

Edit `l2cs_gaze.py` in the `detect_faces()` method, modify the confidence_threshold parameter:

```python
def detect_faces(self, image, confidence_threshold=0.7):  # Lower (e.g., 0.5) for more faces, higher (e.g., 0.9) for stricter
```

Or adjust when calling detect_faces in get_gaze_direction by passing a different threshold.

### Adjust Legacy Method Settings

Edit `gaze_detector.py` in the `GazeDetector.get_gaze_direction()` method (around line 101):

**Gaze thresholds:**

```python
pitch_threshold = (-20, 5)   # Modify as needed
yaw_threshold = (-20, 20)    # Modify as needed
```

**Face detection strictness** (line 62):

```python
minNeighbors=5,  # Higher = stricter (current: 5)
```

**Eye detection strictness** (line 81):

```python
minNeighbors=5,  # Higher = stricter (current: 5)
```

**Eye requirement** (line 87):

```python
if len(eyes) < 2:  # Change to < 1 for more lenient detection
```

**Face aspect ratio** (line 93):

```python
if face_aspect_ratio < 0.65:  # Lower = more lenient (current: 0.65)
```

### Change Visual Markers

Edit the drawing section in `VideoProcessor.process_video()` method (around line 392):

- Bounding box color: `(0, 255, 0)` (BGR format - Green)
- Box thickness: `3` pixels
- Label text size: `0.6`

### Switch Default Method

To change the default gaze method from L2CS to Legacy, edit `gaze_detector.py` line 304:

```python
def __init__(self, yolo_model='yolov8n.pt', confidence=0.5, gaze_method='legacy'):  # Changed from 'l2cs'
```

Or in the argument parser (line 453):

```python
default='legacy',  # Changed from 'l2cs'
```

## License

This project uses open-source libraries with their respective licenses:

- Ultralytics YOLOv8: AGPL-3.0
- OpenCV: Apache 2.0

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## Acknowledgments

- **YOLOv8** by Ultralytics - Human detection
- **L2CS-Net** - "Learning to estimate gaze in unconstrained environments" by Abdelrahman et al.
- **OpenCV** community - Face detection, image processing, and legacy gaze estimation
- **PyTorch** team - Deep learning framework

## References

- L2CS-Net Paper: [https://arxiv.org/abs/2203.03339](https://arxiv.org/abs/2203.03339)
- YOLOv8: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
