import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import warnings
from typing import Union

# Import RoMa modules
from romatch.utils import get_tuple_transform_ops
from romatch.utils.utils import check_rgb, check_not_i16

class RegressionMatcher(nn.Module):
    """
    This class performs regression-based matching using an encoder-decoder architecture.
    """
    def __init__(
        self,
        encoder,
        decoder,
        h=448,
        w=448,
        sample_mode="threshold_balanced",
        upsample_preds=False,
        symmetric=False,
        name=None,
        attenuate_cert=None,
    ):
        super().__init__()
        # Optionally attenuate the certainty scores if needed.
        self.attenuate_cert = attenuate_cert
        # Encoder for feature extraction from input images.
        self.encoder = encoder
        # Decoder to compute correspondences from extracted features.
        self.decoder = decoder
        # Optional name identifier for the matcher instance.
        self.name = name
        # Store resized image width.
        self.w_resized = w
        # Store resized image height.
        self.h_resized = h
        # Set up original image transforms; here only normalization is applied.
        self.og_transforms = get_tuple_transform_ops(resize=None, normalize=True)
        # Define the sampling strategy (e.g., "threshold_balanced").
        self.sample_mode = sample_mode
        # Flag to decide if predictions should be upsampled.
        self.upsample_preds = upsample_preds
        # Define the resolution used when upsampling predictions.
        self.upsample_res = (14*16*6, 14*16*6)
        # Boolean flag to use symmetric matching.
        self.symmetric = symmetric
        # Threshold value used during sampling (optimized after fine-tuning).
        self.sample_thresh = 2.5  # Optimized parameter after fine-tuning (default value in source code is 0.5)

    def get_output_resolution(self):
        """Returns the output resolution based on whether upsampling is enabled."""
        if not self.upsample_preds:
            return self.h_resized, self.w_resized
        else:
            return self.upsample_res

    def extract_backbone_features(self, batch, batched=True, upsample=False):
        """Extracts features from the backbone encoder for both images in the batch."""
        # Get query image (im_A) and source image (im_B) from the batch.
        x_q = batch["im_A"]
        x_s = batch["im_B"]
        if batched:
            # If batched, concatenate along the batch dimension and extract features together.
            X = torch.cat((x_q, x_s), dim=0)
            feature_pyramid = self.encoder(X, upsample=upsample)
        else:
            # Process images separately if not batched.
            feature_pyramid = self.encoder(x_q, upsample=upsample), self.encoder(x_s, upsample=upsample)
        return feature_pyramid

    def fast_kde(self, x, std=0.1, half=True, down=None):
        """
        A fast version of KDE that computes the pairwise squared Euclidean distances
        using matrix multiplications rather than torch.cdist. This should be faster
        than the original if memory permits.

        This version computes:

            dist_sq = ||x||^2 + ||x2||^2.T - 2 * (x @ x2.T)
            scores = exp(-dist_sq / (2*std^2))
            density = scores.sum(dim=-1)

        Args:
            x (torch.Tensor): Input tensor of shape [N, d].
            std (float): Standard deviation for the Gaussian kernel.
            half (bool): Whether to convert x to half precision.
            down (int or None): If provided, use x[::down] as the second argument.

        Returns:
            torch.Tensor: A tensor of shape [N] containing the density estimates.
        """
        # Convert tensor to half precision if required.
        if half:
            x = x.half()

        # Choose the second tensor based on the downsampling parameter.
        x2 = x[::down] if down is not None else x

        # Compute squared norms for each vector.
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  # shape [N, 1]
        x2_norm = (x2 ** 2).sum(dim=1, keepdim=True)  # shape [M, 1]

        # Compute squared Euclidean distances using the formula with broadcasting.
        dist_sq = x_norm + x2_norm.T - 2 * (x @ x2.T)
        # Clamp negative values (caused by floating point errors) to 0.
        dist_sq = torch.clamp(dist_sq, min=0.0)

        # Compute Gaussian kernel scores based on the squared distances.
        scores = torch.exp(-dist_sq / (2 * std**2))
        # Sum the scores to obtain the density estimate for each point.
        density = scores.sum(dim=-1)
        return density

    def sample(self, matches, certainty, num=10000):
        """Sample matches based on certainty and optionally balance them using KDE."""
        # If threshold sampling is enabled, adjust certainty values above the threshold.
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1

        # Flatten the matches and certainty tensors to 2D and 1D respectively.
        matches = matches.reshape(-1, 4)
        certainty = certainty.reshape(-1)

        # Set expansion factor for balanced sampling mode.
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        # Use multinomial sampling to select a subset of good samples.
        good_samples = torch.multinomial(
            certainty,
            num_samples=min(expansion_factor * num, len(certainty)),
            replacement=False
        )
        good_matches = matches[good_samples]
        good_certainty = certainty[good_samples]

        # If not in balanced mode, return the selected matches directly.
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty

        # For balanced mode, compute density estimates using the fast KDE.
        density = self.fast_kde(good_matches, std=0.1, half=True, down=None)

        # Compute sampling probabilities inversely proportional to density.
        p = 1 / (density + 1)
        # For low-density areas, set the probability to a very small number.
        p[density < 10] = 1e-7

        # Resample to obtain balanced samples.
        balanced_samples = torch.multinomial(
            p,
            num_samples=min(num, len(good_certainty)),
            replacement=False
        )
        return good_matches[balanced_samples], good_certainty[balanced_samples]

    def forward(self, batch, batched=True, upsample=False, scale_factor=1):
        """Forward pass to compute correspondences between two images."""
        # Extract feature pyramids for the input batch.
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample=upsample)
        if batched:
            # Split concatenated features into two parts: one for each image.
            f_q_pyramid = {
                scale: f_scale.chunk(2)[0] for scale, f_scale in feature_pyramid.items()
            }
            f_s_pyramid = {
                scale: f_scale.chunk(2)[1] for scale, f_scale in feature_pyramid.items()
            }
        else:
            f_q_pyramid, f_s_pyramid = feature_pyramid
        # Decode the features to generate correspondence predictions.
        corresps = self.decoder(f_q_pyramid,
                               f_s_pyramid,
                               upsample=upsample,
                               **(batch["corresps"] if "corresps" in batch else {}),
                               scale_factor=scale_factor)

        return corresps

    def forward_symmetric(self, batch, batched=True, upsample=False, scale_factor=1):
        """Forward pass for symmetric matching where the order of images is swapped."""
        # Extract features for symmetric matching.
        feature_pyramid = self.extract_backbone_features(batch, batched=batched, upsample=upsample)
        f_q_pyramid = feature_pyramid
        # Swap the order of the two chunks for symmetric processing.
        f_s_pyramid = {
            scale: torch.cat((f_scale.chunk(2)[1], f_scale.chunk(2)[0]), dim=0)
            for scale, f_scale in feature_pyramid.items()
        }
        # Decode the symmetric features to obtain correspondences.
        corresps = self.decoder(f_q_pyramid,
                               f_s_pyramid,
                               upsample=upsample,
                               **(batch["corresps"] if "corresps" in batch else {}),
                               scale_factor=scale_factor)
        return corresps

    def conf_from_fb_consistency(self, flow_forward, flow_backward, th=2):
        """Compute confidence from forward-backward consistency of optical flow."""
        # Assumes flow_forward is of shape (..., H, W, 2)
        has_batch = False
        # Add batch dimension if missing.
        if len(flow_forward.shape) == 3:
            flow_forward, flow_backward = flow_forward[None], flow_backward[None]
        else:
            has_batch = True
        H, W = flow_forward.shape[-3:-1]
        # Adjust threshold relative to image size.
        th_n = 2 * th / max(H, W)
        # Generate a grid of normalized coordinates for the image.
        coords = torch.stack(torch.meshgrid(
            torch.linspace(-1 + 1 / W, 1 - 1 / W, W),
            torch.linspace(-1 + 1 / H, 1 - 1 / H, H), indexing="xy"),
                           dim=-1).to(flow_forward.device)
        # Warp the backward flow using the forward flow field.
        coords_fb = F.grid_sample(
            flow_backward.permute(0, 3, 1, 2),
            flow_forward,
            align_corners=False, mode="bilinear").permute(0, 2, 3, 1)
        # Compute the Euclidean distance between original and warped coordinates.
        diff = (coords - coords_fb).norm(dim=-1)
        # Determine which pixels are consistent within the threshold.
        in_th = (diff < th_n).float()
        if not has_batch:
            in_th = in_th[0]
        return in_th

    def to_pixel_coordinates(self, coords, H_A, W_A, H_B=None, W_B=None):
        """Convert normalized coordinates to pixel coordinates."""
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A)

        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        """Helper function to perform the actual conversion."""
        # Scale normalized coordinates from [-1,1] to pixel coordinate space.
        kpts = torch.stack((W/2 * (coords[..., 0]+1), H/2 * (coords[..., 1]+1)), axis=-1)
        return kpts

    def to_normalized_coordinates(self, coords, H_A, W_A, H_B, W_B):
        """Convert pixel coordinates to normalized coordinates."""
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[..., :2], coords[..., 2:]
        kpts_A = torch.stack((2/W_A * kpts_A[..., 0] - 1, 2/H_A * kpts_A[..., 1] - 1), axis=-1)
        kpts_B = torch.stack((2/W_B * kpts_B[..., 0] - 1, 2/H_B * kpts_B[..., 1] - 1), axis=-1)
        return kpts_A, kpts_B

    def match_keypoints(self, x_A, x_B, warp, certainty, return_tuple=True, return_inds=False):
        """Match keypoints between two images using grid sampling and distance calculations."""
        # Warp image A keypoints into image B space.
        x_A_to_B = F.grid_sample(warp[..., -2:].permute(2, 0, 1)[None], x_A[None, None], align_corners=False, mode="bilinear")[0, :, 0].mT
        # Sample certainty values corresponding to the warped keypoints.
        cert_A_to_B = F.grid_sample(certainty[None, None, ...], x_A[None, None], align_corners=False, mode="bilinear")[0, 0, 0]
        # Compute pairwise Euclidean distances between warped keypoints and keypoints in image B.
        D = torch.cdist(x_A_to_B, x_B)
        # Find indices where the minimal distance condition holds in both directions and meets the certainty threshold.
        inds_A, inds_B = torch.nonzero((D == D.min(dim=-1, keepdim=True).values) * (D == D.min(dim=-2, keepdim=True).values) * (cert_A_to_B[:, None] > self.sample_thresh), as_tuple=True)

        # Return either the keypoints or their indices based on the function parameters.
        if return_tuple:
            if return_inds:
                return inds_A, inds_B
            else:
                return x_A[inds_A], x_B[inds_B]
        else:
            if return_inds:
                return torch.cat((inds_A, inds_B), dim=-1)
            else:
                return torch.cat((x_A[inds_A], x_B[inds_B]), dim=-1)

    @torch.inference_mode()
    def match(
        self,
        im_A_input,
        im_B_input,
        *args,
        batched=False,
        device=None,
    ):
        """Main matching function with inference mode enabled."""
        # Set device to CUDA if available, otherwise use CPU.
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # If the input for image A is a file path, load and convert it.
        if isinstance(im_A_input, (str, os.PathLike)):
            im_A = Image.open(im_A_input)
            check_not_i16(im_A)
            im_A = im_A.convert("RGB")
        else:
            # Verify that the provided image is in RGB.
            check_rgb(im_A_input)
            im_A = im_A_input

        # Repeat the same for image B.
        if isinstance(im_B_input, (str, os.PathLike)):
            im_B = Image.open(im_B_input)
            check_not_i16(im_B)
            im_B = im_B.convert("RGB")
        else:
            check_rgb(im_B_input)
            im_B = im_B_input

        symmetric = self.symmetric
        self.train(False)  # Set model to evaluation mode.
        with torch.no_grad():
            if not batched:
                # For single image pairs, resize and normalize images to expected dimensions.
                b = 1
                w, h = im_A.size
                w2, h2 = im_B.size
                ws = self.w_resized
                hs = self.h_resized

                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True, clahe=False
                )
                im_A, im_B = test_transform((im_A, im_B))
                batch = {"im_A": im_A[None].to(device), "im_B": im_B[None].to(device)}
            else:
                # For batched images, check that both have the same size.
                b, c, h, w = im_A.shape
                b, c, h2, w2 = im_B.shape
                assert w == w2 and h == h2, "For batched images we assume same size"
                batch = {"im_A": im_A.to(device), "im_B": im_B.to(device)}
                if h != self.h_resized or self.w_resized != w:
                    warnings.warn("Model resolution and batch resolution differ, may produce unexpected results")
                hs, ws = h, w
            finest_scale = 1
            # Run the matcher using either symmetric or standard forward pass.
            if symmetric:
                corresps = self.forward_symmetric(batch)
            else:
                corresps = self.forward(batch, batched=True)

            if self.upsample_preds:
                hs, ws = self.upsample_res

            # Optionally adjust certainty using an attenuation factor.
            if self.attenuate_cert:
                low_res_certainty = F.interpolate(
                    corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
                )
                cert_clamp = 0
                factor = 0.5
                low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)

            if self.upsample_preds:
                # Upsample predictions to a finer resolution.
                finest_corresps = corresps[finest_scale]
                torch.cuda.empty_cache()
                test_transform = get_tuple_transform_ops(
                    resize=(hs, ws), normalize=True
                )
                if isinstance(im_A_input, (str, os.PathLike)):
                    im_A, im_B = test_transform(
                        (Image.open(im_A_input).convert('RGB'), Image.open(im_B_input).convert('RGB')))
                else:
                    im_A, im_B = test_transform((im_A_input, im_B_input))

                im_A, im_B = im_A[None].to(device), im_B[None].to(device)
                scale_factor = math.sqrt(self.upsample_res[0] * self.upsample_res[1] / (self.w_resized * self.h_resized))
                batch = {"im_A": im_A, "im_B": im_B, "corresps": finest_corresps}
                if symmetric:
                    corresps = self.forward_symmetric(batch, upsample=True, batched=True, scale_factor=scale_factor)
                else:
                    corresps = self.forward(batch, batched=True, upsample=True, scale_factor=scale_factor)

            # Retrieve flow and certainty outputs from the finest scale.
            im_A_to_im_B = corresps[finest_scale]["flow"]
            certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.attenuate_cert else 0)
            if finest_scale != 1:
                # If necessary, interpolate flow and certainty to match target resolution.
                im_A_to_im_B = F.interpolate(
                    im_A_to_im_B, size=(hs, ws), align_corners=False, mode="bilinear"
                )
                certainty = F.interpolate(
                    certainty, size=(hs, ws), align_corners=False, mode="bilinear"
                )
            # Rearrange the dimensions of the flow tensor.
            im_A_to_im_B = im_A_to_im_B.permute(
                0, 2, 3, 1
            )
            # Create a meshgrid of coordinates for image A.
            im_A_coords = torch.meshgrid(
                (
                    torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                    torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
                ),
                indexing='ij'
            )
            im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))
            im_A_coords = im_A_coords[None].expand(b, 2, hs, ws)
            # Convert certainty logits to probabilities.
            certainty = certainty.sigmoid()  # logits -> probs
            im_A_coords = im_A_coords.permute(0, 2, 3, 1)
            # If the predicted flow is out of range, zero out the corresponding certainty.
            if (im_A_to_im_B.abs() > 1).any() and True:
                wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
                certainty[wrong[:, None]] = 0
            # Clamp flow values to be within the valid range.
            im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
            if symmetric:
                # For symmetric matching, split the flow into two components.
                A_to_B, B_to_A = im_A_to_im_B.chunk(2)
                q_warp = torch.cat((im_A_coords, A_to_B), dim=-1)
                im_B_coords = im_A_coords
                s_warp = torch.cat((B_to_A, im_B_coords), dim=-1)
                warp = torch.cat((q_warp, s_warp), dim=2)
                certainty = torch.cat(certainty.chunk(2), dim=3)
            else:
                # For standard matching, concatenate image A coordinates with flow.
                warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
            if batched:
                return (
                    warp,
                    certainty[:, 0]
                )
            else:
                return (
                    warp[0],
                    certainty[0, 0],
                )

    def visualize_warp(self, warp, certainty, im_A=None, im_B=None,
                     im_A_path=None, im_B_path=None, device="cuda", symmetric=True, save_path=None, unnormalize=False):
        """Visualize the warp result by overlaying warped images with a certainty map."""
        # Determine height and width based on warp dimensions.
        H, W2, _ = warp.shape
        W = W2//2 if symmetric else W2
        # If images are not provided, load them from the given file paths.
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            # Resize and convert images to tensors.
            im_A = im_A.resize((W, H))
            im_B = im_B.resize((W, H))
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        # Warp image B using the predicted flow.
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:, :W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            # Warp image A for symmetric visualization.
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            # Concatenate the warped images side by side.
            warp_im = torch.cat((im_A_transfer_rgb, im_B_transfer_rgb), dim=2)
            white_im = torch.ones((H, 2*W), device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device=device)
        # Blend the warped image with a white background based on the certainty map.
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im
