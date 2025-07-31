import torch
import os

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, hyperparameters):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=hyperparameters['device']))
        print(f"Model loaded from {path}")
        return model
    else:
        print(f"Model file {path} does not exist.")