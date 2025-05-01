# Robust Style Transfer with Transformers

This repository implements and extends the StyTr² architecture for neural style transfer using Vision Transformers. Our work focuses on improving robustness under challenging style inputs—particularly abstract art—by introducing novel separability loss functions.

<img width="654" alt="Screenshot 2025-04-30 at 5 10 34 PM" src="https://github.com/user-attachments/assets/b6951fa6-ad13-4ee3-99e6-888367ecea3f" />

## Overview

Neural style transfer synthesizes images that combine the structural content of one image with the artistic style of another. Transformer-based approaches like StyTr² achieve high fidelity but can produce noisy outputs or struggle with abstract styles. We address these issues by adding two explicit separability losses—**L_sep1** and **L_sep2**—which encourage content and style features to occupy distinct subspaces during training.

## Key Contributions

- **Separability Losses**  
  - **L_sep1**: Content-focused margin loss  
  - **L_sep2**: Joint content-style margin loss  
- **PyTorch Lightning Integration**  
  Modular, reproducible training pipeline with support for multi-GPU.  
- **Empirical Validation**  
  Benchmarked against StyTr² on both abstract and natural image datasets.

## Architecture

- **Vision Transformer Encoders** for content and style  
- **CAPE** (Content-Aware Positional Embedding)  
- **Transformer Decoder** for content-guided stylization  
- **VGG19 Feature Extractor** (frozen) for loss computation

## Datasets

- **Random Image Sample Dataset** (`pankajkumar2002` on Kaggle)  
  3,000 content images (150×150)  
- **Abstract Art Images** (`greg115` on Kaggle)  
  8,145 abstract style references (512×512)  

Both datasets are public, diverse, and high-quality.

## Results

| Model           | Content Loss (Lₙ) | Style Loss (Lₛ) |
|-----------------|-------------------|-----------------|
| StyTr²          | 2.0125            | **1.9512**      |
| **L_sep1**      | **1.9909**        | 1.9630          |
| L_sep2          | 2.0253            | 1.9555          |

- **L_sep1** yields the best content preservation and fastest convergence.  
- **L_sep2** improves resilience to noise under abstract styles.

Here are some example visualizations from our best models:
![final_images](https://github.com/user-attachments/assets/02809a12-d17a-420f-85f6-8d110bf05f3f)


## Setup & Usage

- **Dependencies**  
  All required packages are installed automatically when you open the notebook in Colab. See `requirements.txt` for a static list.

- **Running in Colab**  
  1. Upload `training_pipeline.ipynb` to Colab.
  2. Run all cells—initial cells install dependencies.
  3. Follow the notebook prompts to train or evaluate models.

All functionality is contained within the notebook. Simply open a Jupyter environment and the training pipeline is setup with all necessary hyperparameters.

## Authors

- **Chaitanya Tatipigari**  
  Project lead; proposed separability losses; implemented L_sep1 & L_sep2; PyTorch Lightning setup; VGG19 feature extractor.

- **Alec Kain**  
  Dataset preparation; implemented CAPE, transformer encoders, and patching; explored L1 loss alternatives.

- **Tyler J. Church**  
  Developed transformer and CNN decoders; managed hyperparameter testing; integrated data modules.
