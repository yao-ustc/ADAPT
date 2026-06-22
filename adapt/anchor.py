"""Cross-Attention Visual Anchor refinement (paper Section 4.1).

This module implements:
  1. Attention-map post-processing: normalize, enhance, smooth
  2. Edge halving and point smoothing to suppress border artifacts
  3. Threshold-based cleanup
  4. Final anchor blending onto the original image
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Dict

from .config import ADAPTConfig, default_config
from .scorer import calculate_attention_layer_weights


# ---------------------------------------------------------------------------
# Low-level attention-map transforms
# ---------------------------------------------------------------------------

def normalize(mat: torch.Tensor, method: str = "min") -> torch.Tensor:
    """Normalize tensor to [0, 1].

    Args:
        method: "min" → high values bright; "max" → high values dark.
    """
    if mat.numel() == 0 or mat.max() == mat.min():
        return torch.ones_like(mat) * 0.5
    if method == "max":
        return (mat.max() - mat) / (mat.max() - mat.min())
    elif method == "min":
        return (mat - mat.min()) / (mat.max() - mat.min())
    raise NotImplementedError(f"Unknown normalize method: {method}")


def enhance(mat: torch.Tensor, coe: float = None, cfg: ADAPTConfig = None) -> torch.Tensor:
    """Contrast-enhance attention map via standardization + sigmoid.

    Args:
        coe: enhancement strength (higher = sharper contrast).
    """
    cfg = cfg or default_config
    if coe is None:
        coe = cfg.DEFAULT_ENHANCE_COE
    if mat.std() == 0:
        return torch.ones_like(mat) * 0.5
    mat = (mat - mat.mean()) / mat.std()
    mat = mat * coe
    mat = torch.sigmoid(mat)
    return mat.clamp(0, 1)


def _apply_edge_halving(mask: torch.Tensor, cfg: ADAPTConfig = None) -> torch.Tensor:
    """Reduce attention on image borders to suppress boundary bias."""
    cfg = cfg or default_config
    H, W = mask.shape
    t = cfg.BOUNDARY_PATCH_THICKNESS
    if t * 2 >= H or t * 2 >= W:
        return mask
    bm = torch.zeros_like(mask, dtype=torch.bool)
    bm[:t, :] = True
    bm[-t:, :] = True
    bm[:, :t] = True
    bm[:, -t:] = True
    mask[bm] *= cfg.EDGE_HALVING_FACTOR
    return mask


def _apply_point_smoothing(mask: torch.Tensor, cfg: ADAPTConfig = None) -> torch.Tensor:
    """Diffuse isolated high-attention 'points' with a larger smoothing kernel."""
    cfg = cfg or default_config
    ks = cfg.POINT_SMOOTHING_KERNEL_SIZE
    if ks <= 1 or ks % 2 == 0:
        return mask
    pad = (ks - 1) // 2
    conv = torch.nn.Conv2d(1, 1, ks, padding=pad, padding_mode="replicate",
                           stride=1, bias=False)
    conv.weight.data = torch.ones_like(conv.weight.data) / (ks ** 2)
    conv.to(mask.device)
    inp = mask.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = conv(inp)
    return out.squeeze(0).squeeze(0)


def revise_mask(
    mask: torch.Tensor,
    kernel_size: int = None,
    enhance_coe: float = None,
    apply_threshold: bool = True,
    cfg: ADAPTConfig = None,
) -> torch.Tensor:
    """Refine a raw attention map into a clean, smooth anchor mask.

    Pipeline: normalize → enhance → smooth(conv) → edge-halving →
              point-smoothing → renormalize → threshold → renormalize.

    Args:
        mask: 2D tensor [H, W] of raw attention values.
        kernel_size: smoothing conv kernel size (odd).
        enhance_coe: contrast enhancement coefficient.
        apply_threshold: if True, zero-out low-percentile values.
    Returns:
        Refined 2D tensor [H, W] in [0, 1].
    """
    cfg = cfg or default_config
    if kernel_size is None:
        kernel_size = cfg.DEFAULT_KERNEL_SIZE
    if enhance_coe is None:
        enhance_coe = cfg.DEFAULT_ENHANCE_COE

    mask = normalize(mask, "min")
    mask = enhance(mask, coe=enhance_coe, cfg=cfg)

    # Conv smoothing
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    pad = (kernel_size - 1) // 2
    conv = torch.nn.Conv2d(1, 1, kernel_size, padding=pad, padding_mode="replicate",
                           stride=1, bias=False)
    conv.weight.data = torch.ones_like(conv.weight.data) / (kernel_size ** 2)
    conv.to(mask.device)
    inp = mask.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        mask = conv(inp).squeeze(0).squeeze(0)

    mask = _apply_edge_halving(mask, cfg)
    mask = _apply_point_smoothing(mask, cfg)
    mask = normalize(mask, "min")

    if apply_threshold and mask.numel() > 0 and mask.max() > mask.min():
        thresh = torch.quantile(
            mask.flatten().to(torch.float32), cfg.HEATMAP_THRESHOLD_PERCENTILE
        )
        mask = torch.where(mask < thresh, torch.zeros_like(mask), mask)
        mask = normalize(mask, "min")

    return mask


# ---------------------------------------------------------------------------
# Anchor-to-image blending
# ---------------------------------------------------------------------------

def blend_mask(
    image: Union[str, Path, Image.Image],
    mask: torch.Tensor,
    output_path: Union[str, Path] = None,
    enhance_coe: float = None,
    kernel_size: int = None,
    cfg: ADAPTConfig = None,
) -> Optional[Image.Image]:
    """Overlay a refined attention mask onto the original image.

    Uses transparency blending: image * mask + white * (1 - mask).

    Args:
        image: original image (path or PIL Image).
        mask: 2D attention map [H, W].
        output_path: if set, save the blended image here.
    Returns:
        Blended PIL Image, or None on failure.
    """
    cfg = cfg or default_config
    if enhance_coe is None:
        enhance_coe = cfg.DEFAULT_ENHANCE_COE
    if kernel_size is None:
        kernel_size = cfg.DEFAULT_KERNEL_SIZE

    # Refine the mask
    refined = revise_mask(mask.float(), kernel_size=kernel_size,
                          enhance_coe=enhance_coe, cfg=cfg)
    refined_np = refined.detach().cpu().numpy()

    # Load image
    if isinstance(image, (str, Path)):
        try:
            orig = Image.open(image).convert("RGB")
        except Exception:
            return None
    elif isinstance(image, Image.Image):
        orig = image.convert("RGB")
    else:
        return None

    # Resize mask to image size
    mask_pil = Image.fromarray((refined_np * 255).astype(np.uint8), mode="L")
    mask_pil = mask_pil.resize(orig.size, Image.BICUBIC)
    mask_f = np.array(mask_pil).astype(np.float32) / 255.0

    # Blend: result = image * mask + white * (1 - mask)
    orig_f = np.array(orig).astype(np.float32) / 255.0
    blended = orig_f * mask_f[:, :, np.newaxis] + np.ones_like(orig_f) * (1.0 - mask_f[:, :, np.newaxis])
    result = Image.fromarray((blended * 255).astype(np.uint8))

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

    return result


# ---------------------------------------------------------------------------
# Anchor computation — the main entry point
# ---------------------------------------------------------------------------

def compute_anchor(
    layer_attentions: Dict[int, torch.Tensor],
    cfg: ADAPTConfig = None,
) -> torch.Tensor:
    """Compute the debiased cross-attention visual anchor.

    Args:
        layer_attentions: dict mapping layer_idx → 2D attention map [24, 24].
    Returns:
        Fused 2D anchor tensor [24, 24].
    """
    cfg = cfg or default_config
    weights = calculate_attention_layer_weights(layer_attentions, cfg=cfg)

    patch_dim = cfg.IMAGE_PATCH_DIM
    fused = torch.zeros(patch_dim, patch_dim)
    for layer_idx, attn_map in layer_attentions.items():
        w = weights.get(layer_idx, 0.0)
        if w > 0:
            fused += attn_map.cpu() * w

    if fused.sum() == 0:
        return fused
    return fused


def refine_cross_attention_anchor(
    layer_attentions: Dict[int, torch.Tensor],
    image: Union[str, Path, Image.Image],
    output_path: Union[str, Path] = None,
    cfg: ADAPTConfig = None,
) -> torch.Tensor:
    """Full pipeline: compute fused anchor and refine it into a clean mask.

    This is the top-level entry point for Visual Enhance (paper Section 4.1).

    Returns the refined anchor tensor [24, 24].
    """
    cfg = cfg or default_config
    fused = compute_anchor(layer_attentions, cfg)
    refined = revise_mask(fused.float(), cfg=cfg)

    if output_path is not None and image is not None:
        blend_mask(image, fused.float(),
                   output_path=output_path,
                   cfg=cfg)

    return refined
