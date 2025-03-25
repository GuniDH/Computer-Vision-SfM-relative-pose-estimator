#!/usr/bin/env python
"""
Script to run inference on a set of image pairs using the RoMa model.
"""

import os
import csv
import time
import torch
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from sfm_pose_estimator.models import roma_outdoor


def run_inference(test_dir, test_csv, output_csv, weights_dir=None, save_model=False, save_path=None, visualize=False):
    """
    Run inference on a set of image pairs from a CSV file.
    
    Args:
        test_dir: Directory containing test images
        test_csv: CSV file with image pairs
        output_csv: Path to save output CSV
        weights_dir: Directory containing model weights
        save_model: Whether to save model weights
        save_path: Path to save model weights
        visualize: Whether to visualize matches
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Speed up PyTorch's conv algorithms if input sizes are consistent
    torch.backends.cudnn.benchmark = True
    
    # Check if test files exist
    if not os.path.isfile(test_csv) or not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test files not found: {test_csv} or {test_dir}")
    
    # Load test data
    test_rows = []
    with open(test_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["sample_id"]
            batch_id = row["batch_id"]
            im1_id = row["image_1_id"]
            im2_id = row["image_2_id"]
            test_rows.append((sample_id, batch_id, im1_id, im2_id))
    print(f"Found {len(test_rows)} pairs in {test_csv}.")
    
    # Load model weights
    if weights_dir:
        weights_path = os.path.join(weights_dir, "roma_outdoor.pth")
        dinov2_path = os.path.join(weights_dir, "dinov2_vitl14_pretrain.pth")
        
        if os.path.isfile(weights_path) and os.path.isfile(dinov2_path):
            weights = torch.load(weights_path, map_location=device)
            dinov2_weights = torch.load(dinov2_path, map_location=device)
            roma_model = roma_outdoor(device=device, weights=weights, dinov2_weights=dinov2_weights)
        else:
            print(f"Weights not found in {weights_dir}, downloading from source...")
            roma_model = roma_outdoor(device=device)
    else:
        # Download weights
        roma_model = roma_outdoor(device=device)
    
    # Create visualization directory if needed
    if visualize:
        vis_dir = Path("visualizations")
        vis_dir.mkdir(exist_ok=True)
    
    results = []
    start_inference = time.time()
    
    try:
        # Use inference_mode for speed
        with torch.inference_mode():
            with tqdm(total=len(test_rows), desc="Estimating F matrices") as pbar:
                for (sample_id, scene_name, im1_id, im2_id) in test_rows:
                    try:
                        img1_path = os.path.join(test_dir, scene_name, im1_id + ".jpg")
                        img2_path = os.path.join(test_dir, scene_name, im2_id + ".jpg")
                        
                        # Check existence
                        if not (os.path.isfile(img1_path) and os.path.isfile(img2_path)):
                            print(f"Warning: Images not found for {sample_id}: {img1_path} or {img2_path}")
                            F_est = np.zeros((3, 3), dtype=np.float64)
                        else:
                            # Load sizes
                            W_A, H_A = Image.open(img1_path).size
                            W_B, H_B = Image.open(img2_path).size
                            
                            # Match with RoMa
                            warp, certainty = roma_model.match(img1_path, img2_path, device=device)
                            
                            # Sample matches
                            matches, c = roma_model.sample(warp, certainty)
                            
                            # Convert to pixel coords
                            kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
                            kpts1_np = kpts1.cpu().numpy()
                            kpts2_np = kpts2.cpu().numpy()
                            
                            if kpts1_np.shape[0] < 8:
                                # Not enough matches
                                print(f"Warning: Not enough matches ({kpts1_np.shape[0]}) for {sample_id}")
                                F_est = np.zeros((3, 3), dtype=np.float64)
                            else:
                                # Estimate F with MAGSAC
                                try:
                                    F_est, mask = cv2.findFundamentalMat(
                                        kpts1_np,
                                        kpts2_np,
                                        ransacReprojThreshold=0.7,
                                        method=cv2.USAC_MAGSAC,
                                        confidence=0.999999,
                                        maxIters=10000
                                    )
                                except cv2.error as e:
                                    print(f"MAGSAC error for {sample_id}: {e}")
                                    F_est = None
                                
                                if F_est is None or F_est.shape != (3, 3):
                                    F_est = np.zeros((3, 3), dtype=np.float64)
                                
                                # Visualize matches if requested
                                if visualize:
                                    try:
                                        from sfm_pose_estimator.utils import visualize_matches
                                        
                                        img1 = Image.open(img1_path)
                                        img2 = Image.open(img2_path)
                                        
                                        # Create a visualization of matches
                                        vis_img = visualize_matches(
                                            img1, img2, 
                                            kpts1_np, kpts2_np,
                                            mask=mask.ravel() if mask is not None else None
                                        )
                                        
                                        # Save visualization
                                        vis_path = vis_dir / f"{sample_id}_matches.png"
                                        vis_img.save(vis_path)
                                    except Exception as e:
                                        print(f"Visualization error for {sample_id}: {e}")
                        
                        # Flatten for submission
                        F_str = " ".join(f"{val:e}" for val in F_est.flatten())
                        results.append((sample_id, F_str))
                    
                    except Exception as e:
                        print(f"Error processing pair {sample_id}: {e}")
                        F_est = np.zeros((3, 3), dtype=np.float64)
                        F_str = " ".join(f"{val:e}" for val in F_est.flatten())
                        results.append((sample_id, F_str))
                    
                    finally:
                        pbar.update(1)
        
        end_inference = time.time()
        print(f"Done estimating F for all pairs in {end_inference - start_inference:.2f} seconds.")
        
        # Save F predictions
        output_dir = os.path.dirname(output_csv)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_csv, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["sample_id", "fundamental_matrix"])
            for sample_id, F_str in results:
                writer.writerow([sample_id, F_str])
        
        print(f"Wrote {len(results)} rows to {output_csv}.")
        
        # Save model weights if requested
        if save_model:
            if save_path is None:
                save_path = "roma_full_weights.pth"
            torch.save(roma_model.state_dict(), save_path)
            print(f'Saved model weights to {save_path}')
        
        return results
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on test image pairs")
    parser.add_argument("--test-dir", type=str, required=True,
                      help="Directory containing test images")
    parser.add_argument("--test-csv", type=str, required=True,
                      help="CSV file with image pairs")
    parser.add_argument("--output-csv", type=str, default="submission.csv",
                      help="Path to save output CSV")
    parser.add_argument("--weights-dir", type=str, default=None,
                      help="Directory containing model weights")
    parser.add_argument("--save-model", action="store_true",
                      help="Save model weights after inference")
    parser.add_argument("--save-path", type=str, default=None,
                      help="Path to save model weights")
    parser.add_argument("--visualize", action="store_true",
                      help="Visualize matches")
    
    args = parser.parse_args()
    run_inference(
        args.test_dir,
        args.test_csv,
        args.output_csv,
        args.weights_dir,
        args.save_model,
        args.save_path,
        args.visualize
    )
