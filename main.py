import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from timm.models.vision_transformer import vit_base_patch32_clip_384

from dataset import CelebA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3200, help="number of images to process")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-folder", type=str, default="./images/celeba")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
    dataset = CelebA(args.data_folder, device, 384)
    model = vit_base_patch32_clip_384(pretrained=True)
    model.head = nn.Identity()
    model = model.to(device)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    limit = args.limit // args.batch_size
    with open("./embeddings.txt", "w") as emb_file:
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= limit:
                    break
                imgs, files = batch
                feats = model(imgs).detach().cpu().numpy()
                for j, file in enumerate(files):
                    array_string = " ".join(f"{x:.4f}" for x in feats[j])
                    emb_file.write(f"{file} {array_string}\n")


if __name__ == "__main__":
    main()
