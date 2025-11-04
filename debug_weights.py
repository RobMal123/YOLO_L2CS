"""
Debug script to see exactly which keys match/mismatch between model and weights.
"""

import torch
from l2cs_gaze import L2CSNet
from pathlib import Path

def compare_keys():
    """Compare model keys with weight file keys."""
    
    # Load the model
    print("Creating model...")
    model = L2CSNet(backbone='resnet50', num_bins=90)
    model_keys = set(model.state_dict().keys())
    
    print(f"Model has {len(model_keys)} keys")
    
    # Load the weights
    weights_path = Path('models/L2CSNet_gaze360.pkl')
    print(f"\nLoading weights from {weights_path}...")
    weights = torch.load(weights_path, map_location='cpu')
    weights_keys = set(weights.keys())
    
    print(f"Weights file has {len(weights_keys)} keys")
    
    # Find differences
    missing_in_weights = model_keys - weights_keys
    extra_in_weights = weights_keys - model_keys
    matching_keys = model_keys & weights_keys
    
    print(f"\n{'='*70}")
    print(f"MATCHING KEYS: {len(matching_keys)} / {len(model_keys)}")
    print(f"{'='*70}")
    
    if len(missing_in_weights) > 0:
        print(f"\n⚠ MISSING IN WEIGHTS (model has but weights don't): {len(missing_in_weights)}")
        for key in sorted(missing_in_weights):
            print(f"  - {key}")
    
    if len(extra_in_weights) > 0:
        print(f"\n⚠ EXTRA IN WEIGHTS (weights have but model doesn't): {len(extra_in_weights)}")
        for key in sorted(extra_in_weights)[:20]:  # Show first 20
            print(f"  - {key}")
        if len(extra_in_weights) > 20:
            print(f"  ... and {len(extra_in_weights) - 20} more")
    
    # Show some matching keys
    print(f"\n✓ SAMPLE MATCHING KEYS:")
    for key in sorted(matching_keys)[:10]:
        print(f"  - {key}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Matching: {len(matching_keys)}/{len(model_keys)} ({100*len(matching_keys)/len(model_keys):.1f}%)")
    print(f"  Missing: {len(missing_in_weights)}")
    print(f"  Extra: {len(extra_in_weights)}")
    print(f"{'='*70}")

if __name__ == "__main__":
    compare_keys()

