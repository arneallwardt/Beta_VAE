import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, hyperparameters):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=hyperparameters['device']))
        print(f"Model loaded from {path}")
        return model
    else:
        print(f"Model file {path} does not exist.")

def show_images_grid(imgs_, num_images=25):
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap='Greys_r', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')