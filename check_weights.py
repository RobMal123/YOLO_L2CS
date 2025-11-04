"""
Quick diagnostic script to check L2CS-Net weights file structure.
This helps debug loading issues.
"""

import torch
from pathlib import Path

def check_weights_file(weights_path):
    """Check the structure of a PyTorch weights file."""
    print(f"Checking weights file: {weights_path}")
    print("=" * 70)
    
    try:
        # Load the weights
        state_dict = torch.load(weights_path, map_location='cpu')
        
        print(f"\n✓ File loaded successfully")
        print(f"Type: {type(state_dict)}")
        
        # If it's a dictionary, show the keys
        if isinstance(state_dict, dict):
            # Check if it's wrapped in another dict (common in checkpoints)
            if 'state_dict' in state_dict:
                print("\nNote: Weights are wrapped in 'state_dict' key")
                actual_weights = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                print("\nNote: Weights are wrapped in 'model_state_dict' key")
                actual_weights = state_dict['model_state_dict']
            else:
                actual_weights = state_dict
            
            print(f"\nTotal number of weight keys: {len(actual_weights)}")
            print("\nFirst 20 keys:")
            for i, key in enumerate(list(actual_weights.keys())[:20]):
                shape = actual_weights[key].shape if hasattr(actual_weights[key], 'shape') else 'N/A'
                print(f"  {i+1:2d}. {key:50s} {shape}")
            
            if len(actual_weights) > 20:
                print(f"\n  ... and {len(actual_weights) - 20} more keys")
            
            # Check for specific L2CS keys
            print("\n" + "=" * 70)
            print("Checking for L2CS-Net specific keys:")
            
            key_patterns = {
                'backbone': [k for k in actual_weights.keys() if 'backbone' in k],
                'fc_yaw': [k for k in actual_weights.keys() if 'yaw' in k.lower()],
                'fc_pitch': [k for k in actual_weights.keys() if 'pitch' in k.lower()],
            }
            
            for pattern_name, keys in key_patterns.items():
                if keys:
                    print(f"\n{pattern_name} keys found ({len(keys)}):")
                    for key in keys[:5]:
                        print(f"  - {key}")
                    if len(keys) > 5:
                        print(f"  ... and {len(keys) - 5} more")
                else:
                    print(f"\n{pattern_name} keys: NOT FOUND")
            
        else:
            print(f"\nWarning: Unexpected format (not a dict)")
            
    except Exception as e:
        print(f"\n✗ Error loading file: {e}")
        return False
    
    print("\n" + "=" * 70)
    return True

if __name__ == "__main__":
    # Check for weights in models directory
    models_dir = Path('models')
    
    weight_files = [
        models_dir / 'L2CSNet_gaze360.pkl',
        models_dir / 'l2cs_resnet50.pkl',
    ]
    
    found = False
    for weight_file in weight_files:
        if weight_file.exists():
            found = True
            check_weights_file(weight_file)
            print()
    
    if not found:
        print("No weight files found in models/ directory")
        print("Looking for:")
        for wf in weight_files:
            print(f"  - {wf}")

