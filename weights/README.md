# Weights Directory

This directory is where pre-trained model weights should be stored.

## Required Weight Files

The following weight files are needed for the SfM Pose Estimator:

1. `roma_outdoor.pth`: RoMa model weights optimized for outdoor scenes
2. `dinov2_vitl14_pretrain.pth`: DINOv2 vision transformer weights

## Downloading Weights

You can download the pre-trained weights using the provided script:
You could also train the model yourself with the data, but for my data I found that the pre-trained weights work the best (detailed in main README file)

```bash
python scripts/download_weights.py
```

By default, this will download the outdoor model weights. To download the indoor model weights:

```bash
python scripts/download_weights.py --model-type indoor
```

For the tiny model variant:

```bash
python scripts/download_weights.py --tiny
```

## Manual Download

If the automatic download fails, you can manually download the weights from:

- RoMa weights: https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth
- DINOv2 weights: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth

Place the downloaded files in this directory.

## Note

This directory is ignored by git to avoid committing large weight files. Please download the weights using the provided script or manually as described above.
