"""Task 3, Step 2: reality-check PetNoseDataset.  Run:  python -m utils.check_dataset"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.dataset import IMG_SIZE, PetNoseDataset, denormalize


def check_split(split: str):
    ds = PetNoseDataset(split=split)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    n = 0
    for images, labels, _ in loader:
        assert images.shape[1:] == (3, IMG_SIZE, IMG_SIZE), (
            f"bad image shape {images.shape}"
        )
        assert (labels >= 0).all() and (labels <= IMG_SIZE).all(), (
            "label outside 227x227 frame"
        )
        n += images.shape[0]
    print(f"{split:5s}: {n:,} samples OK")


def plot_samples(n: int = 8, out_path: str = "plots/dataset_check.png"):
    ds = PetNoseDataset(split="train")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat[:n]):
        image, label, _ = ds[i]
        img = denormalize(image).permute(1, 2, 0)  # (3,H,W) -> (H,W,3) for matplotlib
        ax.imshow(img)
        ax.scatter(label[0], label[1], c="red", s=40)  # u = x (column), v = y (row)
        ax.set_title(ds.samples[i][0], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    Path(out_path).parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=100)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    check_split("train")
    check_split("test")
    plot_samples()
