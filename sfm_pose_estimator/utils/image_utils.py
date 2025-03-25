import torch
import numpy as np
from PIL import Image


def check_image_size(image_path, divisor=14):
    """
    Check if image dimensions are divisible by divisor.
    
    Args:
        image_path: Path to the image file
        divisor: Required divisor for dimensions (default 14 for DINOv2)
        
    Returns:
        tuple: (width, height, is_valid)
    """
    with Image.open(image_path) as img:
        width, height = img.size
    
    is_valid = (width % divisor == 0) and (height % divisor == 0)
    return width, height, is_valid


def resize_to_valid(image, divisor=14):
    """
    Resize an image to dimensions divisible by divisor.
    
    Args:
        image: PIL Image
        divisor: Required divisor for dimensions
        
    Returns:
        PIL.Image: Resized image
    """
    width, height = image.size
    new_width = (width // divisor) * divisor
    new_height = (height // divisor) * divisor
    
    # Only resize if necessary
    if new_width != width or new_height != height:
        return image.resize((new_width, new_height), Image.LANCZOS)
    return image


def tensor_to_pil(tensor, unnormalize=False):
    """
    Convert a PyTorch tensor to a PIL Image.
    
    Args:
        tensor: PyTorch tensor of shape [C, H, W] or [H, W, C]
        unnormalize: Whether to unnormalize from [-1, 1] to [0, 1]
        
    Returns:
        PIL.Image: Converted image
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first batch item
    
    if tensor.dim() == 2:
        # Grayscale
        tensor = tensor.unsqueeze(0)
    
    if tensor.shape[0] in [1, 3, 4]:
        # [C, H, W] format, convert to [H, W, C]
        tensor = tensor.permute(1, 2, 0)
    
    if unnormalize:
        tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
    
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and then to PIL
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    
    np_array = (tensor.detach().numpy() * 255).astype(np.uint8)
    
    # Handle single-channel vs multi-channel
    if np_array.shape[-1] == 1:
        np_array = np_array.squeeze(-1)
        return Image.fromarray(np_array, mode='L')
    elif np_array.shape[-1] == 3:
        return Image.fromarray(np_array, mode='RGB')
    elif np_array.shape[-1] == 4:
        return Image.fromarray(np_array, mode='RGBA')
    else:
        return Image.fromarray(np_array)


def visualize_matches(image1, image2, keypoints1, keypoints2, matches=None, mask=None, margin=10):
    """
    Visualize keypoint matches between two images.
    
    Args:
        image1: First image (PIL or np.array)
        image2: Second image (PIL or np.array)
        keypoints1: Keypoints in first image (N x 2)
        keypoints2: Keypoints in second image (N x 2)
        matches: Matches indices or None if keypoints are already matched
        mask: Optional mask for valid/invalid matches
        margin: Margin between images
        
    Returns:
        PIL.Image: Visualization of matches
    """
    import cv2
    
    # Convert PIL to numpy if needed
    if isinstance(image1, Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, Image.Image):
        image2 = np.array(image2)
    
    # Convert grayscale to RGB if needed
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
    
    # Create a blank image that fits both images side-by-side with a margin
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    height = max(h1, h2)
    width = w1 + w2 + margin
    
    vis_img = np.zeros((height, width, 3), dtype=np.uint8)
    vis_img[:h1, :w1] = image1
    vis_img[:h2, w1+margin:] = image2
    
    # Draw keypoints and matches
    if matches is None:
        # Assume keypoints are already matched one-to-one
        matches = np.arange(len(keypoints1))
        matches = np.stack([matches, matches], axis=1)
    
    if mask is None:
        mask = np.ones(len(matches), dtype=bool)
    
    # Colors for matches (green for inliers, red for outliers)
    color_inlier = (0, 255, 0)
    color_outlier = (0, 0, 255)
    
    # Draw lines between matches
    for i, (idx1, idx2) in enumerate(matches):
        pt1 = tuple(map(int, keypoints1[idx1]))
        pt2 = tuple(map(int, [keypoints2[idx2][0] + w1 + margin, keypoints2[idx2][1]]))
        
        color = color_inlier if mask[i] else color_outlier
        cv2.line(vis_img, pt1, pt2, color, 1, cv2.LINE_AA)
    
    # Draw keypoints
    for pt in keypoints1:
        cv2.circle(vis_img, tuple(map(int, pt)), 3, (255, 0, 0), -1)
    
    offset = np.array([w1 + margin, 0])
    for pt in keypoints2:
        pt_with_offset = tuple(map(int, pt + offset))
        cv2.circle(vis_img, pt_with_offset, 3, (255, 0, 0), -1)
    
    return Image.fromarray(vis_img)
