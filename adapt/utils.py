"""Utility functions for image I/O and tensor conversion."""

from PIL import Image
import torch
import torchvision.transforms as T
from pathlib import Path
from typing import Union


def read_image(p: Union[str, Path]) -> Image.Image:
    """Read an image file and return as RGB PIL Image."""
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    """Convert a torch tensor to PIL Image."""
    if t.ndim == 2:
        return T.ToPILImage()(t.unsqueeze(0))
    return T.ToPILImage()(t)


def resize_mask(mask: Image.Image, target: Image.Image,
                method: Image.Resampling = Image.BICUBIC) -> Image.Image:
    """Resize a mask PIL Image to match the target image dimensions."""
    return mask.resize(target.size, method)
