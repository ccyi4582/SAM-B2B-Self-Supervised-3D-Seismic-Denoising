# SAM-B2B: Self-Supervised 3D Seismic Denoising

This repository contains the official PyTorch implementation of **SAM-B2B**, a Spatially Adaptive Multiscale Block-to-Block network for self-supervised 3D seismic data denoising.

## Features
- **3D Block-to-Block (B2B) Blind-spot Mechanism**: SAM-B2B uses 3D block-to-block masks to disrupt spatial noise dependencies.
- **Adaptive Multiscale Masking**: Adaptive multiscale masking dynamically fits local texture complexity.
- **Adaptive Residual Fusion**: Dynamic fusion integrates hard scale selection with residual attention.

## Requirements
- Python 3.8+
- PyTorch 1.10+
- NumPy, OpenCV, scikit-image, tqdm

Install dependencies:
```bash
pip install -r requirements.txt
