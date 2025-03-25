import torch
import torch.nn as nn
import warnings
from typing import Union
import math

from romatch.models.matcher import ConvRefiner, CosKernel, GP, Decoder
from romatch.models.transformer import Block, TransformerDecoder, MemEffAttention
from romatch.models.encoders import CNNandDinov2

from sfm_pose_estimator.models.regression_matcher import RegressionMatcher

# Dictionary containing URLs for pretrained model weights
weight_urls = {
    "romatch": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
}


def roma_model(resolution, upsample_preds, device=None, weights=None, dinov2_weights=None, amp_dtype: torch.dtype=torch.float16, **kwargs):
    """
    Create the complete Roma model by assembling encoder, decoder, and additional modules.
    
    Args:
        resolution: Tuple of (height, width) for input resolution
        upsample_preds: Whether to upsample predictions
        device: Device to place model on
        weights: Pre-trained weights for Roma model
        dinov2_weights: Pre-trained weights for DINOv2
        amp_dtype: Data type for automatic mixed precision
        
    Returns:
        RegressionMatcher: The assembled Roma model
    """
    # Suppress specific warnings related to deprecated storage types
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    
    # Define dimensions for global projection and features
    gp_dim = 512
    feat_dim = 512
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 64
    
    # Build a transformer-based coordinate decoder
    coordinate_decoder = TransformerDecoder(
        nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]),
        decoder_dim,
        cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp=True,
        pos_enc=False,)
    
    dw = True
    hidden_blocks = 8
    kernel_size = 5
    displacement_emb = "linear"
    disable_local_corr_grad = True

    # Construct convolutional refiners for various scales
    conv_refiner = nn.ModuleDict(
        {
            "16": ConvRefiner(
                2 * 512+128+(2*7+1)**2,
                2 * 512+128+(2*7+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=128,
                local_corr_radius=7,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "8": ConvRefiner(
                2 * 512+64+(2*3+1)**2,
                2 * 512+64+(2*3+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=64,
                local_corr_radius=3,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "4": ConvRefiner(
                2 * 256+32+(2*2+1)**2,
                2 * 256+32+(2*2+1)**2,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=32,
                local_corr_radius=2,
                corr_in_other=True,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "2": ConvRefiner(
                2 * 64+16,
                128+16,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=16,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
            "1": ConvRefiner(
                2 * 9 + 6,
                24,
                2 + 1,
                kernel_size=kernel_size,
                dw=dw,
                hidden_blocks=hidden_blocks,
                displacement_emb=displacement_emb,
                displacement_emb_dim=6,
                amp=True,
                disable_local_corr_grad=disable_local_corr_grad,
                bn_momentum=0.01,
            ),
        }
    )
    
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    
    # Create a Gaussian Process module for scale "16"
    gp16 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    gps = nn.ModuleDict({"16": gp16})
    
    # Define projection layers for various scales
    proj16 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1), nn.BatchNorm2d(512))
    proj8 = nn.Sequential(nn.Conv2d(512, 512, 1, 1), nn.BatchNorm2d(512))
    proj4 = nn.Sequential(nn.Conv2d(256, 256, 1, 1), nn.BatchNorm2d(256))
    proj2 = nn.Sequential(nn.Conv2d(128, 64, 1, 1), nn.BatchNorm2d(64))
    proj1 = nn.Sequential(nn.Conv2d(64, 9, 1, 1), nn.BatchNorm2d(9))
    proj = nn.ModuleDict({
        "16": proj16,
        "8": proj8,
        "4": proj4,
        "2": proj2,
        "1": proj1,
        })
    
    displacement_dropout_p = 0.0
    gm_warp_dropout_p = 0.0
    
    # Assemble the decoder using the coordinate decoder, GP modules, projection layers, and convolutional refiners
    decoder = Decoder(coordinate_decoder,
                     gps,
                     proj,
                     conv_refiner,
                     detach=True,
                     scales=["16", "8", "4", "2", "1"],
                     displacement_dropout_p=displacement_dropout_p,
                     gm_warp_dropout_p=gm_warp_dropout_p)

    # Initialize the encoder using CNN and Dinov2 modules
    encoder = CNNandDinov2(
        cnn_kwargs=dict(
            pretrained=False,
            amp=True),
        amp=True,
        use_vgg=True,
        dinov2_weights=dinov2_weights,
        amp_dtype=amp_dtype,
    )
    
    h, w = resolution
    symmetric = True
    attenuate_cert = True
    sample_mode = "threshold_balanced"
    
    # Create an instance of RegressionMatcher with all the assembled components
    matcher = RegressionMatcher(encoder, decoder, h=h, w=w, upsample_preds=upsample_preds,
                              symmetric=symmetric, attenuate_cert=attenuate_cert, 
                              sample_mode=sample_mode, **kwargs).to(device)
    matcher.load_state_dict(weights)
    return matcher


def roma_outdoor(device, weights=None, dinov2_weights=None, coarse_res: Union[int, tuple[int, int]]=560, 
                upsample_res: Union[int, tuple[int, int]]=864, amp_dtype: torch.dtype=torch.float16):
    """
    Convenience function to load and configure the Roma model for outdoor scenes.
    
    Args:
        device: Device to place model on
        weights: Pre-trained weights for Roma model
        dinov2_weights: Pre-trained weights for DINOv2
        coarse_res: Coarse resolution for initial processing
        upsample_res: Resolution for upsampling
        amp_dtype: Data type for automatic mixed precision
        
    Returns:
        RegressionMatcher: The configured Roma model for outdoor scenes
    """
    # Convert resolutions to tuples if they are provided as integers
    if isinstance(coarse_res, int):
        coarse_res = (coarse_res, coarse_res)
    if isinstance(upsample_res, int):
        upsample_res = (upsample_res, upsample_res)

    # Use float32 precision if running on CPU
    if str(device) == 'cpu':
        amp_dtype = torch.float32

    # Ensure the coarse resolution dimensions are multiples of 14 as required by the backbone
    assert coarse_res[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert coarse_res[1] % 14 == 0, "Needs to be multiple of 14 for backbone"

    # Load model weights from URL if not provided
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(weight_urls["romatch"]["outdoor"],
                                                   map_location=device)
    if dinov2_weights is None:
        dinov2_weights = torch.hub.load_state_dict_from_url(weight_urls["dinov2"],
                                                         map_location=device)
    # Build the Roma matcher model
    model = roma_model(resolution=coarse_res, upsample_preds=True,
               weights=weights, dinov2_weights=dinov2_weights, device=device, amp_dtype=amp_dtype)
    # Set the upsample resolution
    model.upsample_res = upsample_res
    print(f"Using coarse resolution {coarse_res}, and upsample res {model.upsample_res}")
    return model
