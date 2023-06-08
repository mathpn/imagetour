import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
from PIL import Image
from sklearn.decomposition import PCA


def save_html(idx: list[int], files: list[str]):
    with open("./tsp_order.html", "w") as html:
        for i in idx:
            html.write(f"<img src='./{files[i]}' height=40px />")


def save_comparison(emb, tsp_order: list[int], files: list[str], n_samples: int, n_neighbors: int):
    pca = PCA(n_components=1)
    emb_pca = pca.fit_transform(emb).ravel()
    pca_order = np.argsort(emb_pca)
    samples = random.sample(range(emb.shape[0]), k=n_samples)
    fig = plt.figure(figsize=(6, 6))
    grid = ImageGrid(fig, 111, nrows_ncols=(2 * n_samples, n_neighbors + 1), axes_pad=0)
    i = 0
    for sample in samples:
        pca_idx = np.where(pca_order == sample)[0][0]
        pca_segment = pca_order[pca_idx : pca_idx + n_neighbors + 1]
        tsp_idx = tsp_order.index(sample)
        tsp_segment = tsp_order[tsp_idx : tsp_idx + n_neighbors + 1]
        for segment, name in ((tsp_segment, "TSP"), (pca_segment, "PCA")):
            for j, idx in enumerate(segment):
                ax = grid[i]
                if j == 0:
                    ax.set_ylabel(name, size="large")
                file = files[idx]
                img = Image.open(file)
                ax.imshow(img)
                i += 1
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
    plt.tight_layout()
    plt.savefig("./img_samples.png")
    print("saved image to ./img_samples.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", action="store_true")
    parser.add_argument(
        "--n-samples", type=int, default=3, help="number of sample images to use as starting points"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=5,
        help="number of forward nearest-neighbors to show per sample",
    )
    args = parser.parse_args()

    with open("./LKH-3.0.8/imagetour.out", "r") as f:
        lkh = f.readlines()

    with open("./embeddings.txt", "r") as emb_f:
        emb = emb_f.readlines()

    files = [x.split(" ")[0] for x in emb]
    emb = [list(map(float, x.split(" ")[1:])) for x in emb]
    emb = np.array(emb)

    idx = lkh[6:-2]
    idx = [int(x) - 1 for x in idx]
    if args.html:
        save_html(idx, files)
    else:
        save_comparison(emb, idx, files, args.n_samples, args.n_neighbors)


if __name__ == "__main__":
    main()
