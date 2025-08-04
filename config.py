# YOU ONLY NEED TO CHANGE THESE 2 PARAMETERS TO EXPERIMENT WITH THIS REPOSITORY
# make sure to restart the kernel used for the notebook after making changes here
# --------------------------------------------------------------------------------
dataset_name = "3d_shapes" # dsprites / 3d_shapes
vae_mode = "beta" # default / beta
# --------------------------------------------------------------------------------


# naming & paths
champion_model_name = 'champion'
default_vae_load_model_dir = f'default_VAE_{dataset_name}'
beta_vae_load_model_dir = f'beta_VAE_{dataset_name}'
load_model_dir = default_vae_load_model_dir if vae_mode == 'default' else beta_vae_load_model_dir
results_path = "./results"

# general params
img_width_height = 64
no_channels = 1 if dataset_name == 'dsprites' else 3
disentanglement_range = [-4, 4]
disentanglement_steps = 16

# hyperparameters VAEs
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if vae_mode == 'beta':
    beta = 10 if dataset_name == "dsprites" else 32 # changed 4 -> 10
else: 
    beta = 1
latent_dim = 5 if dataset_name == "dsprites" else 6 # changed: 10 -> 5

train_val_split = 0.8
lr = 1e-4 # proposed value by FactorVAE paper
batch_size = 64 # proposed value by FactorVAE paper
num_epochs = 1000
log_interval = 100
early_stopping_rounds = 5
early_stopping_min_delta = 1e-2

# hyperparameters majority vote classifier
train_test_split = 0.8
majority_vote_classifier_samples_per_factor = 2000
majority_vote_classifier_all_val_batch_size = 512