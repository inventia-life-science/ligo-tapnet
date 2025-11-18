#!/usr/bin/env python3
"""Test script to import and instantiate TAPIR and Depth Anything V2 models."""

import os
import sys
import torch

# Add Depth-Anything-V2/metric_depth to Python path for imports
depth_anything_path = os.path.expanduser("~/repos/Depth-Anything-V2/metric_depth")
if os.path.exists(depth_anything_path):
    sys.path.insert(0, depth_anything_path)
else:
    print(f"Warning: Depth-Anything-V2 path not found: {depth_anything_path}")

print("=" * 60)
print("Testing model imports and instantiation")
print("=" * 60)

# Test 1: Import TAPIR model
print("\n1. Importing TAPIR model...")
try:
    from tapnet.torch_tapir import tapir_model
    print("   ✓ TAPIR model imported successfully")
    
    # Instantiate TAPIR model
    print("   Creating TAPIR model...")
    tapir_checkpoint = "tapnet/checkpoints/causal_bootstapir_checkpoint.pt"
    
    if not os.path.exists(tapir_checkpoint):
        print(f"   ✗ TAPIR checkpoint not found: {tapir_checkpoint}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {device}")
        
        model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
        model.load_state_dict(torch.load(tapir_checkpoint, map_location=device))
        model = model.to(device).eval()
        torch.set_grad_enabled(False)
        
        print("   ✓ TAPIR model instantiated and loaded successfully")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
except Exception as e:
    print(f"   ✗ Failed to import/instantiate TAPIR: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Import Depth Anything V2 model
print("\n2. Importing Depth Anything V2 model...")
try:
    from depth_anything_v2.dpt import DepthAnythingV2
    print("   ✓ DepthAnythingV2 imported successfully")
    
    # Instantiate Depth Anything V2 model
    print("   Creating Depth Anything V2 model...")
    depth_checkpoint = os.path.join(
        depth_anything_path, 
        "checkpoints", 
        "depth_anything_v2_metric_hypersim_vits.pth"
    )
    
    if not os.path.exists(depth_checkpoint):
        print(f"   ✗ Depth checkpoint not found: {depth_checkpoint}")
        print(f"   Looking for alternative checkpoints...")
        checkpoint_dir = os.path.join(depth_anything_path, "checkpoints")
        if os.path.exists(checkpoint_dir):
            available = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            print(f"   Available checkpoints: {available}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {device}")
        
        # Model config for vits (from run.py)
        model_config = {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384],
            'max_depth': 20.0  # Default from run.py
        }
        
        depth_model = DepthAnythingV2(**model_config)
        depth_model.load_state_dict(torch.load(depth_checkpoint, map_location='cpu'))
        depth_model = depth_model.to(device).eval()
        
        print("   ✓ Depth Anything V2 model instantiated and loaded successfully")
        print(f"   Model parameters: {sum(p.numel() for p in depth_model.parameters()):,}")
        print(f"   Checkpoint: {depth_checkpoint}")
        
except Exception as e:
    print(f"   ✗ Failed to import/instantiate Depth Anything V2: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)


