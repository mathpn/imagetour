import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torchvision.utils import save_image


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = Flowers102("./data", download=True, transform=transform)
print(dataset)

model = resnet18(ResNet18_Weights.DEFAULT)
# model.fc = nn.Linear(512, 10)
model.fc = nn.Identity()
device = torch.device('cuda')
model = model.to(device)
print(model)

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
# criterion = nn.CrossEntropyLoss()
# optim = torch.optim.Adam(model.parameters())

# for epoch in range(1):
#     print(epoch)
#     for imgs, labels in data_loader:
#         optim.zero_grad()
#         imgs = imgs.to(device).expand(-1, 3, -1, -1)
#         labels = labels.to(device)
#         out = model(imgs)
#         loss = criterion(out, labels)
#         loss.backward()
#         optim.step()


# model.fc = nn.Identity()

os.makedirs("./images", exist_ok=True)
all_feats = []
with open("./embeddings.txt", "w") as emb_file:
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            imgs, labels = batch
            imgs = imgs.to(device)
            # imgs = imgs.expand(-1, 3, -1, -1)
            feats = model(imgs).detach().cpu().numpy()
            for j in range(len(imgs)):
                img = imgs[j]
                img_id = i * 32 + j
                save_image(img, f"./images/img_{img_id}.jpg")
                array_string = ' '.join(f"{x:.4f}" for x in feats[j])
                emb_file.write(f"img_{img_id} {array_string}\n")

