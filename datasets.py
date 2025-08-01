import torch
from torch.utils.data import Dataset

class DSpritesDataset(Dataset):
    def __init__(self, imgs):
        # imgs: numpy array (N, 64, 64)
        self.imgs = torch.from_numpy(imgs).float()  # to float32
        self.imgs = self.imgs.unsqueeze(1)  # (N, 1, 64, 64) → add channel dim

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]


class ShapesDataset(Dataset):
    def __init__(self, imgs):
        # imgs: numpy array (N, 64, 64, 3)
        self.imgs = torch.from_numpy(imgs / 255.).float() # to float32
        self.imgs = self.imgs.permute(0, 3, 1, 2) # (N, 3, 64, 64) → create channel first tensor

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx]