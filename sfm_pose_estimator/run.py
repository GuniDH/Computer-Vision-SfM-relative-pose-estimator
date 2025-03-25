import os
import csv
import time
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from sfm_pose_estimator.models.roma import roma_outdoor

def main():
    """
    Main function to run the SfM pose estimator on a test dataset.
    
    This function:
    1. Loads the Roma model
    2. Processes image pairs from the test dataset
    3. Estimates the fundamental matrix for each pair
    4. Saves the results to a submission file
    """
    # Speed up PyTorch's conv algorithms if input sizes are consistent
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define file paths
    TEST_DIR = os.getenv("TEST_DIR", "/content/dataset/test_images")
    TEST_CSV = os.getenv("TEST_CSV", "/content/dataset/test.csv")
    SUBMISSION_CSV = os.getenv("SUBMISSION_CSV", "/content/submission.csv")
    WEIGHTS_PTH = os.getenv("WEIGHTS_PTH", "/content/roma_full_weights.pth")

    if not os.path.isfile(TEST_CSV) or not os.path.isdir(TEST_DIR):
        print(f"Test Files not found!")
        return

    # Load test data
    test_rows = []
    with open(TEST_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["sample_id"]
            batch_id = row["batch_id"]
            im1_id = row["image_1_id"]
            im2_id = row["image_2_id"]
            test_rows.append((sample_id, batch_id, im1_id, im2_id))
    print(f"Found {len(test_rows)} pairs in {TEST_CSV}.")

    # Load the Roma model
    roma_model = roma_outdoor(device=device)

    results = []
    start_inference = time.time()

    try:
        # Use inference_mode for speed
        with torch.inference_mode():
            with tqdm(total=len(test_rows), desc="Estimating F on test pairs") as pbar:
                for (sample_id, scene_name, im1_id, im2_id) in test_rows:
                    try:
                        img1_path = os.path.join(TEST_DIR, scene_name, im1_id + ".jpg")
                        img2_path = os.path.join(TEST_DIR, scene_name, im2_id + ".jpg")

                        # Check existence
                        if not (os.path.isfile(img1_path) and os.path.isfile(img2_path)):
                            F_est = np.zeros((3, 3), dtype=np.float64)
                        else:
                            # Load sizes
                            W_A, H_A = Image.open(img1_path).size
                            W_B, H_B = Image.open(img2_path).size

                            # Match with RoMa
                            warp, certainty = roma_model.match(img1_path, img2_path, device=device)

                            # Sample
                            matches, c = roma_model.sample(warp, certainty)

                            # Convert to pixel coords
                            kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
                            kpts1_np = kpts1.cpu().numpy()
                            kpts2_np = kpts2.cpu().numpy()

                            if kpts1_np.shape[0] < 8:
                                # Not enough matches
                                F_est = np.zeros((3, 3), dtype=np.float64)
                            else:
                                # Estimate F with MAGSAC
                                '''
                                MAGSAC is a RANSAC variation. I'll shortly explain how RANSAC works to find F:
                                Given at least 8 point correspondences, 8 points will be used to calculate the 
                                hypothesis F with SVD, the other correspondences will be checked to be inliers this way:
                                given one correspondence (x,x') we will use F to get the epipolar line in which x' 
                                lies which should be Fx. If the distance from x' to Fx is less than a specified 
                                threshold, the correspondence is an inlier. After some iterations or when we reached 
                                a specified threshold of inliers for an hypothesis, we will return the F matrix (model)
                                with the maximal amount of inliers.
                                '''
                                try:
                                    F_est, mask = cv2.findFundamentalMat(
                                        kpts1_np,
                                        kpts2_np,
                                        ransacReprojThreshold=0.7,  # Changed from 0.2 after fine-tuning and optimization
                                        method=cv2.USAC_MAGSAC,
                                        confidence=0.999999,
                                        maxIters=10000
                                    )
                                except cv2.error:
                                    F_est = None
                                if F_est is None or F_est.shape != (3, 3):
                                    F_est = np.zeros((3, 3), dtype=np.float64)

                        # Flatten for submission
                        F_str = " ".join(f"{val:e}" for val in F_est.flatten())
                        results.append((sample_id, F_str))

                    except Exception as e:  # Usually happens when VRAM is full
                        print(f"Error processing pair {sample_id}: {e}")
                        F_est = np.zeros((3, 3), dtype=np.float64)
                        F_str = " ".join(f"{val:e}" for val in F_est.flatten())
                        results.append((sample_id, F_str))

                    finally:
                        pbar.update(1)

        end_inference = time.time()
        print(f"Done estimating F for all pairs in {end_inference - start_inference:.2f} seconds.")

        # Save F predictions and model weights
        with open(SUBMISSION_CSV, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["sample_id", "fundamental_matrix"])
            for sample_id, F_str in results:
                writer.writerow([sample_id, F_str])

        print(f"Wrote {len(results)} rows to {SUBMISSION_CSV}.")
        torch.save(roma_model.state_dict(), WEIGHTS_PTH)
        print('Saved model weights')
        
        # Optional: Download the submission file if running in Colab
        try:
            from google.colab import files
            files.download(SUBMISSION_CSV)
        except ImportError:
            pass  # Not running in Colab

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
