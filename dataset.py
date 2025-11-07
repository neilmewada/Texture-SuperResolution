import os
import random
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# Optional: silence PIL's "decompression bomb" warning for big textures
Image.MAX_IMAGE_PIXELS = None


def _list_images(root: str, exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp")) -> List[str]:
    paths = []
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(dp, f))
    return paths


def _ensure_divisible_by(img: Image.Image, s: int) -> Image.Image:
    """Crop right/bottom so (W,H) are divisible by s."""
    w, h = img.size
    w2 = (w // s) * s
    h2 = (h // s) * s
    if w2 == w and h2 == h:
        return img
    return img.crop((0, 0, w2, h2))


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor [C,H,W] in [0,1]."""
    arr = np.asarray(img).astype(np.float32) / 255.0  # H,W,3
    arr = np.transpose(arr, (2, 0, 1))               # 3,H,W
    return torch.from_numpy(arr)


class AmbientCG_FSRCNN_Dataset(Dataset):
    """
    Dataset for FSRCNN_model (RGB, scale ∈ {2,3,4}).

    Returns:
        lr: [3, h, w]  (low-res)
        hr: [3, H, W]  (high-res), where H = h*scale, W = w*scale

    Args:
        root_dir: folder containing ambientCG textures (1K, square or rectangular).
        scale: 2 | 3 | 4 (must match FSRCNN_model(scale)).
        mode: 'patch' or 'full'.
        hr_patch_size: HR patch size (only for 'patch' mode). Must be divisible by scale.
        patches_per_image: how many patches to draw per image pseudo-epoch (patch mode).
        augment: random flips / 90° rotations applied on HR before downsampling.
        file_limit: optionally cap number of source files (for quick experiments).
        seed: RNG seed for reproducible patch order.
    """
    def __init__(
        self,
        root_dir: str,
        scale: int = 4,
        mode: str = "patch",
        hr_patch_size: int = 128,
        patches_per_image: int = 16,
        augment: bool = True,
        file_limit: Optional[int] = None,
        seed: int = 42,
    ):
        assert mode in ("patch", "full")
        assert scale in (2, 3, 4), "FSRCNN_model supports scales 2, 3, or 4"
        if mode == "patch":
            assert hr_patch_size % scale == 0, "hr_patch_size must be divisible by scale"

        self.root_dir = root_dir
        self.scale = scale
        self.mode = mode
        self.hr_patch_size = hr_patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment

        files = _list_images(root_dir)
        if file_limit is not None:
            files = files[:file_limit]
        if len(files) == 0:
            print(f"[WARN] No images found under: {root_dir}")
        self.files = files

        # Build a deterministic index map for patch mode
        self.idx_map: List[Tuple[int, int]] = []
        if self.mode == "patch":
            for img_idx in range(len(self.files)):
                for k in range(self.patches_per_image):
                    self.idx_map.append((img_idx, k))
            random.Random(seed).shuffle(self.idx_map)

    def __len__(self):
        return len(self.idx_map) if self.mode == "patch" else len(self.files)

    @staticmethod
    def _random_hr_patch(img: Image.Image, size: int) -> Image.Image:
        w, h = img.size
        if w < size or h < size:
            # upsize small images so we can crop a patch
            img = img.resize((max(w, size), max(h, size)), Image.Resampling.BICUBIC)
            w, h = img.size
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        return img.crop((x, y, x + size, y + size))

    @staticmethod
    def _augment_4(img: Image.Image) -> Image.Image:
        """Random H/V flips and 90° rotations (keeps tiling-friendly transforms)."""
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        # 0/90/180/270
        k = random.randint(0, 3)
        if k:
            img = img.rotate(90 * k, expand=False)
        return img

    def __getitem__(self, idx):
        if self.mode == "patch":
            img_idx, _ = self.idx_map[idx]
            path = self.files[img_idx]
            hr = Image.open(path).convert("RGB")
            hr = self._random_hr_patch(hr, self.hr_patch_size)
            hr = _ensure_divisible_by(hr, self.scale)
            if self.augment:
                hr = self._augment_4(hr)
        else:
            path = self.files[idx]
            hr = Image.open(path).convert("RGB")
            hr = _ensure_divisible_by(hr, self.scale)
            # no augmentation in 'full' mode by default

        # Create LR by downsampling HR by `scale`
        hr_w, hr_h = hr.size
        lr_size = (hr_w // self.scale, hr_h // self.scale)
        lr = hr.resize(lr_size, Image.Resampling.BICUBIC)

        # -> tensors [3,H,W] and [3,h,w]
        hr_t = _to_tensor(hr)
        lr_t = _to_tensor(lr)

        return lr_t, hr_t, os.path.basename(path)