import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision import transforms
from torchvision.utils import save_image

dataset = MNIST("./data", download=True, train=True, transform=transforms.ToTensor())
print(dataset)

model = mobilenet_v2(MobileNet_V2_Weights.DEFAULT)
model.classifier = nn.Identity()
device = torch.device('cuda')
model = model.to(device)
print(model)

data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

os.makedirs("./images", exist_ok=True)
all_feats = []
with open("./embeddings.txt", "w") as emb_file:
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            imgs = imgs.expand(-1, 3, -1, -1)
            feats = model(imgs).detach().cpu().numpy()
            for j in range(len(imgs)):
                img = imgs[j]
                img_id = i * 32 + j
                save_image(img, f"./images/img_{img_id}.jpg")
                array_string = ' '.join(f"{x:.4f}" for x in feats[j])
                emb_file.write(f"img_{img_id} {array_string}\n")

            if i >= 10:
                break
