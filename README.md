# Beta-VAE for MNIST Disentanglement

This project is part of the Smart Graphics module at the University of Bremen, Summer Semester 2025. The goal is to implement and train a Beta-Variational Autoencoder (Beta-VAE) to achieve a disentangled feature space using the MNIST dataset.

## Project Overview

The Beta-VAE is an extension of the traditional Variational Autoencoder that introduces a hyperparameter β to control the trade-off between reconstruction quality and the degree of disentanglement in the latent space. This project aims to:

- Implement a Beta-VAE architecture
- Train the model on the MNIST dataset
- Analyze the disentanglement properties of the learned latent space
- Visualize and interpret the learned features

## Requirements

- Python 3.12+
- PyTorch
- torchvision
- matplotlib
- numpy

## Project Structure

```
Beta_VAE/
├── data/                  # Directory for MNIST dataset
├── models/               # Model architecture definitions
├── utils/               # Utility functions
├── data_exploration.ipynb  # Initial data exploration
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv beta_vae_venv
source beta_vae_venv/bin/activate  # On Unix/macOS
# or
.\beta_vae_venv\Scripts\activate  # On Windows
```

2. Install required packages:
```bash
pip install torch torchvision matplotlib numpy
```

## Usage

1. Data Exploration:
   - Run the `vae.ipynb` notebook to train a traditional VAE and visualize its feature space. 

2. Model Training:
   - [To be implemented] Training scripts and procedures will be added

3. Analysis:
   - [To be implemented] Analysis tools and visualization methods will be added

## Background

### Beta-VAE
The Beta-VAE is a modification of the traditional VAE that introduces a hyperparameter β to control the trade-off between reconstruction quality and the degree of disentanglement in the latent space. The objective function is:

L(θ, φ; x, z, β) = E_qφ(z|x)[log pθ(x|z)] - β * KL(qφ(z|x) || p(z))

where:
- θ and φ are the parameters of the decoder and encoder respectively
- x is the input data
- z is the latent representation
- β controls the strength of the KL divergence term

### Disentanglement
Disentanglement refers to the property where each dimension of the latent space corresponds to a single, interpretable factor of variation in the data. In the context of MNIST, this could mean:
- One dimension controlling the digit's thickness
- Another dimension controlling the digit's slant
- A third dimension controlling the digit's position

## Future Work

- Implement the Beta-VAE architecture
- Add training scripts
- Create visualization tools for the latent space
- Analyze the degree of disentanglement
- Compare different β values and their effects

## License

This project is part of an academic course and is intended for educational purposes.

## Contact

[Your Name] - [Your Email]