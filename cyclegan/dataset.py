"""
Unpaired image dataset for CycleGAN training.

Domain A (synthetic) and Domain B (real) live in independent train / test
directories under data/. Pairs are random — at every training step we sample
one image from each domain independently. The dataset's length is
max(len(A), len(B)) so neither side dominates: whichever side is shorter
gets cycled through.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


SUPPORTED_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def _list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    """
    Image transform pipeline.

    Train:  resize → random horizontal flip → ToTensor → Normalize to [-1, 1].
    Eval:   resize → ToTensor → Normalize to [-1, 1].

    Normalising to [-1, 1] is required because the generator's final tanh
    activation outputs in that range.
    """
    ops: list = [
        transforms.Resize((image_size, image_size), antialias=True),
    ]
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(ops)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Map a [-1, 1] tensor back to [0, 1] for saving / display."""
    return (tensor.clamp(-1.0, 1.0) + 1.0) * 0.5


class UnpairedImageDataset(Dataset):
    """
    Loads one image from each of two domain directories per __getitem__.

    Domain A is iterated in order (modulo length); Domain B is sampled
    randomly so each pass through A sees a fresh cross-product. This matches
    the standard CycleGAN training recipe.
    """

    def __init__(
        self,
        root_a: str | os.PathLike,
        root_b: str | os.PathLike,
        image_size: int = 256,
        train: bool = True,
    ):
        self.root_a = Path(root_a)
        self.root_b = Path(root_b)
        self.files_a = _list_images(self.root_a)
        self.files_b = _list_images(self.root_b)

        if not self.files_a:
            raise FileNotFoundError(f"No images found in domain A directory: {self.root_a}")
        if not self.files_b:
            raise FileNotFoundError(f"No images found in domain B directory: {self.root_b}")

        self.image_size = image_size
        self.train = train
        self.transform = build_transform(image_size, train=train)

    def __len__(self) -> int:
        return max(len(self.files_a), len(self.files_b))

    def _load(self, path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert('RGB')
            return self.transform(im)

    def __getitem__(self, idx: int) -> dict:
        path_a = self.files_a[idx % len(self.files_a)]
        if self.train:
            # Random pick from B so A and B aren't implicitly paired by index.
            path_b = random.choice(self.files_b)
        else:
            path_b = self.files_b[idx % len(self.files_b)]

        return {
            'A': self._load(path_a),
            'B': self._load(path_b),
            'A_path': str(path_a),
            'B_path': str(path_b),
        }


class SingleDomainDataset(Dataset):
    """
    Loads images from a single directory — used by test.py (inference) where
    we only need source-domain images, no pairing.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        image_size: int = 256,
    ):
        self.root = Path(root)
        self.files = _list_images(self.root)
        if not self.files:
            raise FileNotFoundError(f"No images found in: {self.root}")
        self.transform = build_transform(image_size, train=False)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        with Image.open(path) as im:
            im = im.convert('RGB')
            tensor = self.transform(im)
        return {'image': tensor, 'path': str(path), 'name': path.stem}
