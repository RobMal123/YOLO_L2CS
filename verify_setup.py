"""
Setup Verification Script
Checks if all dependencies are installed correctly and tests basic functionality.
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError as e:
        print(f"✗ {package_name} is NOT installed: {e}")
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available (Device: {torch.cuda.get_device_name(0)})")
            return True
        else:
            print("⚠ CUDA is NOT available (CPU mode only)")
            return False
    except:
        return False

def main():
    print("=" * 60)
    print("YOLO Gaze Detection System - Setup Verification")
    print("=" * 60)
    print()
    
    # Check Python version
    print("Python Version:")
    version = sys.version_info
    print(f"  {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print("  ✓ Python version is compatible (3.8+)")
    else:
        print("  ✗ Python version too old (need 3.8+)")
        return False
    print()
    
    # Check required packages
    print("Checking Required Packages:")
    all_ok = True
    
    packages = [
        ('cv2', 'opencv-python'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('ultralytics', 'ultralytics'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('gdown', 'gdown'),
    ]
    
    for module, package in packages:
        if not check_import(module, package):
            all_ok = False
    
    print()
    
    # Check CUDA
    print("Checking GPU Support:")
    check_cuda()
    print()
    
    # Check custom modules
    print("Checking Custom Modules:")
    if check_import('l2cs_gaze'):
        print("  ✓ L2CS-Net module is available")
    else:
        print("  ✗ L2CS-Net module not found")
        all_ok = False
    
    if check_import('gaze_detector'):
        print("  ✓ Gaze detector module is available")
    else:
        print("  ✗ Gaze detector module not found")
        all_ok = False
    
    print()
    
    # Test L2CS initialization
    print("Testing L2CS-Net Initialization:")
    try:
        from l2cs_gaze import L2CSGazeEstimator
        print("  Initializing L2CS-Net (this may take a moment)...")
        estimator = L2CSGazeEstimator()
        print("  ✓ L2CS-Net initialized successfully")
        estimator.cleanup()
        print("  ✓ L2CS-Net cleanup successful")
    except Exception as e:
        print(f"  ✗ L2CS-Net initialization failed: {e}")
        all_ok = False
    
    print()
    
    # Final status
    print("=" * 60)
    if all_ok:
        print("✓ All checks passed! System is ready to use.")
        print()
        print("Quick start:")
        print("  python gaze_detector.py --input video.mp4 --output output.mp4")
    else:
        print("✗ Some checks failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        print()
        print("For GPU support (recommended):")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("=" * 60)
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

