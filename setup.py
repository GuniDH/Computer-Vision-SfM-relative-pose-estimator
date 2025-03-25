from setuptools import setup, find_packages

setup(
    name="sfm_pose_estimator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "tqdm>=4.60.0",
        "pandas>=1.3.0",
        "zipfile36>=0.1.0",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "sfm-download-weights=scripts.download_weights:main",
            "sfm-inference=scripts.run_inference:main",
        ],
    },
    author="Guni Deyo Haness",
    description="Structure from Motion (SfM) Relative Pose Estimator using RoMa",
    keywords="computer-vision, structure-from-motion, pose-estimation, roma",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
