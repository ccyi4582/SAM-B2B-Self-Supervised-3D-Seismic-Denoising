# SAM-B2B: Self-Supervised 3D Seismic Denoising

This repository contains the official PyTorch implementation of **SAM-B2B**, a Spatially Adaptive Multiscale Block-to-Block network for self-supervised 3D seismic data denoising.

## Features
- **3D Block-to-Block (B2B) Blind-spot Mechanism**: Effectively disrupts spatial noise dependencies in 3D volumes.
- **Spatial Complexity Analysis**: Dynamically evaluates local texture complexity using 3D gradients.
- **Adaptive Multiscale Masking**: Allocates appropriate mask scales based on local geological structures.
- **Adaptive Residual Fusion**: Softly integrates features from different scales using a lightweight attention mechanism.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy, OpenCV, scikit-image, tqdm

Install dependencies:
```bash
pip install -r requirements.txt
