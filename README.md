# HybridSR: Hybrid Window Self-Attention & State Space Models for SISR

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**HybridSR** is a novel deep learning architecture for Single Image Super-Resolution (SISR) that systematically unifies the three dominant paradigms of image feature extraction into a single, cohesive network. This repository contains the official inference code and pre-trained models from the Final Year Project Thesis: *"HybridSR: A Hybrid Window Self-Attention and State Space Model Architecture for Single Image Super-Resolution"*.

## 🧠 Architecture Overview

HybridSR leverages a three-scale design philosophy, allocating specific architectural blocks to the spatial scales at which they are most effective, without employing spatial downsampling:

1. **Local Detail (RRDBs)**: 12 Residual-in-Residual Dense Blocks (RRDB) extract pixel-precise local textures using 3×3 convolutions.
2. **Mid-Range Structure (Swin Transformers)**: Swin Transformer blocks (W-MSA & SW-MSA) compute self-attention across 8×8 windows to capture repeating structural patterns and mid-range relationships.
3. **Global Coherence (SS2D Mamba)**: Four-directional Selective State Space (SS2D) scans enforce image-wide structural consistency with linear computational complexity ($\mathcal{O}(HW)$).

These modules are wrapped in a **Long Skip Connection (LSC)** and fed into a **PixelShuffle** upsampling head to directly reconstruct the high-resolution image.

## 📂 Repository Structure

```
HybridSR/
├── src/
│   ├── upscale_worker2.py      # 2x Super-Resolution inference script
│   ├── upscale_worker2gen.py   # 2x SR generalized model script
│   ├── upscale_worker4.py      # 4x Super-Resolution inference script
│   └── upscale_worker4gen.py   # 4x SR generalized model script
├── weights/
│   ├── true-generalized-2x-best_val37.6036_epoch85.weights.h5
│   ├── true-generalized-4x-best_val31.6411_epoch103.weights.h5
│   └── (Additional Mamba specific weights)
├── HybridSR.pdf                # Full thesis detailing the architecture & ablations
└── requirements.txt            # Python dependencies
```

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Inference
To upscale an image using the pre-trained HybridSR models, run the corresponding worker script from the `src` directory and point it to the weights in the `weights` directory.

**For 2x Upscaling:**
```bash
python src/upscale_worker2.py path/to/input.png path/to/output.png weights/true-generalized-2x-best_val37.6036_epoch85.weights.h5
```

**For 4x Upscaling:**
```bash
python src/upscale_worker4.py path/to/input.png path/to/output.png weights/true-generalized-4x-best_val31.6411_epoch103.weights.h5
```

## 📊 Performance (Set5)
On the standard Set5 benchmark (bicubic $\times 2$ degradation), HybridSR achieves a highly competitive Y-channel PSNR of **>37.8 dB**, combining the precision of CNNs, the structural reasoning of Vision Transformers, and the global coherence of State Space Models.

## 🎓 About
This project was developed as a Bachelor's Graduation Thesis. For in-depth details on the hyperparameter choices, ablation studies, and mathematical formulations of the SS2D Mamba blocks, please consult the provided `HybridSR.pdf` document.
