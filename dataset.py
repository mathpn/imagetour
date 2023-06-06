"""
Helper to load the CelebA images.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class CelebA(Dataset):
    def __init__(self, data_path: str, device: torch.device, img_size: int = 384):
        self.data_path = data_path
        self.img_size = img_size
        self.device = device
        self.images = sorted(x for x in os.listdir(self.data_path) if x.endswith(".jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = f"{self.data_path}/{self.images[idx]}"
        img = Image.open(img_path)
        img = self._transform_img(img).to(self.device)
        return img, img_path

    def _transform_img(self, img):
        img = F.to_tensor(np.array(img))
        img = F.resize(img, [self.img_size, self.img_size], antialias=True)
        return img
