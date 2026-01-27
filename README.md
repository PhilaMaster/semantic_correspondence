# Semantic Correspondence with Vision Foundation Models

This repository implements semantic correspondence using state-of-the-art vision foundation models. The project evaluates and fine-tunes models like DINOv2, DINOv3, and Segment Anything Model (SAM) for dense matching tasks, enabling precise keypoint matching across semantically similar images.

## Overview

Semantic correspondence aims to find matching points between two images of the same or similar objects/scenes, even when they differ in viewpoint, pose, or appearance. This project:

- **Evaluates** pre-trained vision foundation models (DINOv2, DINOv3, SAM) on semantic correspondence benchmarks
- **Fine-tunes** these models with contrastive learning for improved matching performance
- **Implements** multiple matching strategies including argmax and windowed soft-argmax
- **Supports** ensemble evaluation combining predictions from multiple models
- **Benchmarks** performance on standard datasets: SPair-71K, PF-Pascal, and PF-Willow

## Key Features

### Models
- **DINOv2** (ViT-Base/14): Self-supervised vision transformer with 14×14 patches
- **DINOv3** (ViT-Base/16): Advanced vision transformer with storage tokens
- **SAM** (Segment Anything Model): Vision encoder from Meta's segmentation model

### Matching Strategies
- **Argmax**: Hard matching by selecting the most similar patch
- **Windowed Soft-Argmax**: Soft matching with temperature-controlled smoothing in a local window

### Evaluation Metrics
- **PCK (Percentage of Correct Keypoints)**: Measures correspondence accuracy at multiple thresholds (0.05, 0.1, 0.2)
- Normalized by image diagonal (PF-Pascal, PF-Willow) or bounding box (SPair-71K)

### Training & Evaluation
- **Zero-shot evaluation**: Test pre-trained models without fine-tuning
- **Fine-tuning**: Adapt models using cross-entropy loss on correspondence task
- **Hyperparameter search**: Optimize temperature and window size for soft-argmax
- **Ensemble evaluation**: Combine multiple models for improved performance

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.12+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/PhilaMaster/semantic_correspondence.git
cd semantic_correspondence
```

2. Install dependencies:
```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install other requirements
pip install numpy pandas scikit-learn jupyter
```

3. Set up model weights:
```bash
# Download pre-trained weights for DINOv2, DINOv3, and SAM
# Place weights in respective model directories:
# - models/dinov2/weights/
# - models/dinov3/weights/
# - models/segment_anything/weights/
```

4. Download benchmark datasets:
- **SPair-71K**: Place in `SPair71k/` directory
- **PF-Pascal**: Place in `pf_pascal/` directory  
- **PF-Willow**: Place in `pf_willow/` directory

## Usage

### Zero-Shot Evaluation

Evaluate pre-trained models without fine-tuning:

```bash
# DINOv2 on PF-Pascal
python training_free_evaluation.py

# DINOv3 on SPair-71K
python training_free_evaluation_faster.py

# SAM on benchmark dataset
python training_free_evaluation_SAM.py
```

### Fine-Tuning

Fine-tune models on the correspondence task:

```bash
# Fine-tune DINOv2
python finetuning/finetune_dinov2.py

# Fine-tune DINOv3
python finetuning/finetune_dinov3.py

# Fine-tune SAM encoder
python finetuning/finetune_SAM.py
```

Fine-tuning configuration:
- Freeze most layers, unfreeze last N blocks
- Cross-entropy loss with temperature scaling
- Contrastive learning on keypoint correspondences

### Ensemble Evaluation

Combine predictions from multiple models:

```bash
python ensemble_evaluate.py
```

This loads DINOv2, DINOv3, and SAM models and averages their similarity scores for improved matching.

### Hyperparameter Search

Optimize soft-argmax parameters:

```bash
python grid_search_hyperparams.py
```

Searches over:
- Window size K: {3, 5, 7, 9, 11}
- Temperature T: {0.05, 0.1, 0.2, 0.5, 1.0}

### Evaluation

Evaluate fine-tuned models:

```bash
python evaluate.py
```

Results are saved to timestamped directories with detailed metrics per image and per keypoint.

## Project Structure

```
semantic_correspondence/
├── models/                          # Model implementations
│   ├── dinov2/                     # DINOv2 model and weights
│   ├── dinov3/                     # DINOv3 model and weights
│   └── segment_anything/           # SAM model and weights
├── finetuning/                      # Fine-tuning scripts and experiments
│   ├── finetune_dinov2.py
│   ├── finetune_dinov3.py
│   ├── finetune_SAM.py
│   └── blocks_comparison/          # Ablation studies
├── SPair71k/                        # SPair-71K dataset
├── pf_pascal/                       # PF-Pascal dataset utilities
├── pf_willow/                       # PF-Willow dataset utilities
├── results_*/                       # Evaluation results (timestamped)
├── helper_functions.py              # Feature extraction utilities
├── matching_strategies.py           # Matching algorithms (argmax, soft-argmax)
├── pck.py                          # PCK metric computation
├── evaluate.py                      # Main evaluation script
├── ensemble_evaluate.py             # Multi-model ensemble evaluation
├── grid_search_hyperparams.py       # Hyperparameter optimization
└── training_free_evaluation*.py     # Zero-shot evaluation scripts
```

## Evaluation Metrics

### PCK (Percentage of Correct Keypoints)

A keypoint is considered correct if the Euclidean distance between predicted and ground truth locations is below a threshold:

- **Threshold values**: 0.05, 0.1, 0.2 (relative to normalization factor)
- **Normalization**: 
  - SPair-71K: Bounding box max(width, height)
  - PF-Pascal/PF-Willow: Image diagonal

### Results Format

Results include:
- Per-image PCK at each threshold
- Per-keypoint accuracy
- Inference time
- Detailed metrics saved as JSON and CSV

## Datasets

### SPair-71K
- 71,000+ image pairs across 18 object categories
- Dense keypoint annotations
- Diverse viewpoints and poses

### PF-Pascal
- Based on PASCAL VOC 2011
- Image pairs with keypoint annotations
- Focused on object categories

### PF-Willow  
- Smaller benchmark dataset
- Challenging viewpoint variations

## Implementation Details

### Feature Extraction
- Images resized to model-specific sizes (518×518 for DINOv2, 512×512 for DINOv3/SAM)
- Dense feature maps extracted from patch tokens
- Features normalized before matching

### Matching Process
1. Extract dense features from source and target images
2. For each source keypoint, compute similarity with all target patches
3. Find best match using argmax or windowed soft-argmax
4. Convert patch coordinates back to pixel coordinates
5. Compute PCK against ground truth

### Fine-Tuning Strategy
- Freeze early layers, unfreeze last N transformer blocks
- Cross-entropy loss on correspondence classification
- Temperature-scaled softmax for soft targets
- Learning rate: 1e-4 to 1e-5
- Temperature: 10-15 for training


## Acknowledgments

This project builds upon:
- **DINOv2**: Meta AI's self-supervised vision transformer
- **DINOv3**: Advanced vision transformer architecture
- **SAM**: Meta's Segment Anything Model
- **SPair-71K**: Semantic correspondence benchmark dataset
- **PF-Pascal & PF-Willow**: Proposal Flow benchmark datasets
