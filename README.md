# Beta VAE - Disentanglement Representation Learning

This repository contains the implementation and evaluation of Beta Variational Autoencoders (β-VAEs) for disentanglement representation learning. This project was developed as part of the "Smart Graphics" university course, where I investigate and compare the disentanglement capabilities of β-VAEs against standard VAEs.

## 📋 Overview

The project implements and evaluates two types of Variational Autoencoders:
- **Standard VAE**: Baseline implementation with β = 1
- **Beta VAE**: Enhanced implementation with β > 1 for improved disentanglement

The research focuses on analyzing how different β values affect the learned latent representations and their disentanglement properties across different datasets.

## 🏗️ Project Structure

```
Beta_VAE/
├── config.py                          # Centralized configuration file
├── train_vae.ipynb                    # Main training notebook
├── quantitative_evaluation.ipynb      # Quantitative analysis
├── qualitative_evaluation.ipynb       # Qualitative analysis
├── models.py                          # VAE model implementations
├── datasets.py                        # Dataset loading utilities
├── requirements.txt                   # Python dependencies
├── utils/
│   ├── download_data.py              # Data download script
│   └── utils.py                      # Utility functions
├── data/                             # Dataset storage
├── results/                          # Training results and models
└── README.md                         # This file
```

## ⚙️ Quick Start

### 1. System Requirements

- **Python**: 3.12.x recommended
- **CUDA**: Compatible system recommended for GPU acceleration

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Beta_VAE
   ```

2. **Install dependencies:**
   
   **For CUDA-compatible systems:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **For non-CUDA systems:**
   ```bash
   # Install PyTorch CPU version first
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # Then install remaining dependencies manually
   pip install numpy matplotlib seaborn pandas h5py tqdm python-dotenv dropbox
   ```

### 3. Data Setup

1. **Get Dropbox Access Token:**
   - Contact the repository maintainer to obtain a Dropbox access token
   - The token is required to download the datasets

2. **Download Data:**
   ```bash
   python utils/download_data.py
   ```

### 4. Configuration

The project uses a centralized configuration system in `config.py`. **You only need to modify the first two variables:**

```python
dataset_name = "dsprites" # dsprites / 3d_shapes
vae_mode = "beta" # default / beta
```

**⚠️ Important:** After changing these parameters, **restart the Jupyter kernel** to ensure the changes take effect.

### 5. Training and Evaluation

1. **Open the training notebook:**
   ```bash
   jupyter notebook train_vae.ipynb
   ```

2. **Run the cells sequentially** to train your VAE model

3. **Evaluate results:**
   - Use `quantitative_evaluation.ipynb` for metrics and analysis
   - Use `qualitative_evaluation.ipynb` for visual inspection

## 🔬 Research Context

This project investigates disentanglement representation learning capabilities of β-VAEs compared to standard VAEs. The research is part of the "Smart Graphics" university course, focusing on:

- **Disentanglement Analysis**: How well different latent factors are separated
- **Representation Quality**: Comparison between β-VAE and standard VAE representations
- **Dataset Performance**: Evaluation on dSprites and 3D Shapes datasets

## 📊 Datasets

The project supports two datasets:
- **dSprites**: 2D shapes with 5 latent factors (shape, scale, rotation, x, y)
- **3D Shapes**: 3D objects with 6 latent factors (floor hue, wall hue, object hue, scale, shape, orientation)

This project is developed for educational and research purposes as part of the university course "Smart Graphics".