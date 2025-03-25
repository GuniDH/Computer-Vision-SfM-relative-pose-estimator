# Data Directory

This directory is where datasets should be placed for processing with the SfM Pose Estimator.
*I can't provide the data I worked with because it was a part of Kaggle competition so I can't download the dataset.*

## Dataset Structure

Datasets should be organized in the following structure:

```
data/
├── train/
│   ├── scene1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── scene2/
│   │   └── ...
│   └── ...
├── test/
│   ├── scene1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── scene2/
│   │   └── ...
│   └── ...
├── train.csv
└── test.csv
```

## CSV Format

The CSV files should follow this format:

```
sample_id,batch_id,image_1_id,image_2_id
pair_0,scene_1,image_a,image_b
pair_1,scene_1,image_c,image_d
...
```

Where:
- `sample_id`: Unique identifier for the image pair
- `batch_id`: Scene/batch identifier
- `image_1_id`: Filename (without extension) of the first image
- `image_2_id`: Filename (without extension) of the second image

## Note

This directory is ignored by git to avoid committing large dataset files. Please download and place your datasets here manually.
