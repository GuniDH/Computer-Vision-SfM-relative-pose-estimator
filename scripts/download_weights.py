#!/usr/bin/env python
"""
Script to download pre-trained weights for the RoMa model.
"""

import os
import torch
import argparse
from pathlib import Path
from sfm_pose_estimator.models import weight_urls

def download_weights(output_dir, model_type="outdoor", use_tiny=False):
    """
    Download pre-trained weights for the RoMa model.
    
    Args:
        output_dir: Directory to save the weights
        model_type: Type of model ('outdoor' or 'indoor')
        use_tiny: Whether to use the tiny model variant
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which weights to download
    model_key = "tiny_roma_v1" if use_tiny else "romatch"
    
    if model_type not in weight_urls[model_key]:
        raise ValueError(f"Model type '{model_type}' not available for {'tiny' if use_tiny else 'standard'} model. "
                        f"Available types: {list(weight_urls[model_key].keys())}")
    
    # Download Roma weights
    roma_url = weight_urls[model_key][model_type]
    roma_filename = f"roma_{'tiny_' if use_tiny else ''}{model_type}.pth"
    roma_path = output_dir / roma_filename
    
    print(f"Downloading Roma weights from {roma_url}...")
    roma_weights = torch.hub.load_state_dict_from_url(
        roma_url, map_location=torch.device('cpu'), progress=True
    )
    torch.save(roma_weights, roma_path)
    print(f"Saved Roma weights to {roma_path}")
    
    # Download DINOv2 weights
    dinov2_url = weight_urls["dinov2"]
    dinov2_path = output_dir / "dinov2_vitl14_pretrain.pth"
    
    print(f"Downloading DINOv2 weights from {dinov2_url}...")
    dinov2_weights = torch.hub.load_state_dict_from_url(
        dinov2_url, map_location=torch.device('cpu'), progress=True
    )
    torch.save(dinov2_weights, dinov2_path)
    print(f"Saved DINOv2 weights to {dinov2_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pre-trained weights for RoMa model")
    parser.add_argument("--output-dir", type=str, default="weights", 
                      help="Directory to save the weights")
    parser.add_argument("--model-type", type=str, choices=["outdoor", "indoor"], default="outdoor",
                      help="Type of model to download")
    parser.add_argument("--tiny", action="store_true", 
                      help="Download tiny model variant")
    
    args = parser.parse_args()
    download_weights(args.output_dir, args.model_type, args.tiny)
