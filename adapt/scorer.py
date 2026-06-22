"""Attention-map quality scoring functions.

These implement the three complementary criteria from the paper (Section 4.1):
  - Spectral Consistency (S_spec)  → frequency_match_score
  - Spatial Smoothness (S_smooth)  → smoothness_score
  - Adaptive Focus (S_focus)       → concentration_score

Plus two auxiliary scores for robust layer fusion:
  - Boundary penalty
  - Layer-position prior
"""

import torch
import torch.nn.functional as F
from typing import Dict

from .config import ADAPTConfig, default_config


def calculate_boundary_penalty_score(
    attn_map: torch.Tensor,
    boundary_thickness: int = None,
    penalty_ratio_threshold: float = None,
    cfg: ADAPTConfig = None,
) -> float:
    """Penalize attention that concentrates on image borders.

    Returns 1.0 (no penalty) to 0.0 (maximum penalty).
    """
    cfg = cfg or default_config
    if boundary_thickness is None:
        boundary_thickness = cfg.BOUNDARY_PATCH_THICKNESS
    if penalty_ratio_threshold is None:
        penalty_ratio_threshold = cfg.BOUNDARY_PENALTY_THRESHOLD_RATIO

    H, W = attn_map.shape
    if boundary_thickness * 2 >= H or boundary_thickness * 2 >= W:
        return 1.0

    # Collect boundary values (4 edges)
    top = attn_map[:boundary_thickness, :]
    bottom = attn_map[-boundary_thickness:, :]
    left = attn_map[:, :boundary_thickness]
    right = attn_map[:, -boundary_thickness:]

    boundary_vals = torch.cat([
        top.flatten(), bottom.flatten(),
        left[:, boundary_thickness:-boundary_thickness].flatten(),
        right[:, boundary_thickness:-boundary_thickness].flatten(),
    ])
    center_vals = attn_map[boundary_thickness:-boundary_thickness,
                           boundary_thickness:-boundary_thickness]

    eps = 1e-6
    mean_b = boundary_vals.mean().item()
    mean_c = center_vals.mean().item()

    if mean_c < eps and mean_b < eps:
        return 1.0
    if mean_c > eps:
        ratio = mean_b / mean_c
    else:
        ratio = float("inf") if mean_b > eps else 1.0

    if ratio > penalty_ratio_threshold:
        score = 1.0 - min(1.0, (ratio - penalty_ratio_threshold) * cfg.BOUNDARY_PENALTY_STRENGTH)
        return max(0.0, score)
    return 1.0


def calculate_frequency_match_score(
    attn_map: torch.Tensor,
    layer_idx: int,
    total_layers: int = None,
    radial_cutoff_ratio: float = None,
    cfg: ADAPTConfig = None,
) -> float:
    """Score how well the attention FFT matches the expected layer-wise profile.

    Early layers → higher high-freq energy; later layers → lower high-freq.
    Implements S_spec from the paper (Eq. 1-2).
    """
    cfg = cfg or default_config
    if total_layers is None:
        total_layers = cfg.TOTAL_ATTENTION_LAYERS
    if radial_cutoff_ratio is None:
        radial_cutoff_ratio = cfg.FFT_RADIAL_CUTOFF_RATIO

    if attn_map.numel() == 0 or attn_map.max() == attn_map.min():
        return 0.5

    # Normalize and FFT (force float32 — FFT doesn't support float16)
    norm = (attn_map.float() - attn_map.float().min()) / (attn_map.float().max() - attn_map.float().min() + 1e-6)
    fft = torch.fft.fft2(norm)
    fft_s = torch.fft.fftshift(fft)
    power = torch.abs(fft_s) ** 2

    H, W = power.shape
    cy, cx = H // 2, W // 2
    y = torch.arange(H, device=attn_map.device) - cy
    x = torch.arange(W, device=attn_map.device) - cx
    Y, X = torch.meshgrid(y, x, indexing="ij")
    radii = torch.sqrt(X.float() ** 2 + Y.float() ** 2)

    max_r = torch.sqrt(torch.tensor(cy ** 2 + cx ** 2, dtype=torch.float32, device=attn_map.device))
    cutoff = max_r * radial_cutoff_ratio

    high_mask = radii > cutoff
    total_energy = power.sum()
    if total_energy < 1e-6:
        return 0.5

    high_ratio = (power * high_mask).sum() / (total_energy + 1e-6)

    # Ideal: shallow layers have ~80% high-freq, deep layers ~20%
    norm_layer = layer_idx / (total_layers - 1 + 1e-6)
    ideal_high = 0.8 * (1.0 - norm_layer) + 0.2 * norm_layer

    return float(1.0 - torch.abs(high_ratio - ideal_high).item())


def calculate_concentration_score(attn_map: torch.Tensor) -> float:
    """Score attention concentration via inverse normalized entropy.

    Implements S_focus from the paper (Section 4.1.1).
    Higher = more concentrated; lower = more diffuse.
    """
    if attn_map.numel() == 0 or attn_map.max() == attn_map.min():
        return 0.0

    norm = attn_map / (attn_map.sum() + 1e-6)
    entropy = -(norm * torch.log(norm + 1e-9)).sum()
    max_entropy = torch.log(
        torch.tensor(attn_map.numel(), dtype=torch.float32, device=attn_map.device) + 1e-9
    )
    if max_entropy < 1e-6:
        return 0.0
    return float((1.0 - entropy / max_entropy).clamp(0, 1).item())


def calculate_smoothness_score(
    attn_map: torch.Tensor,
    sobel_x: torch.Tensor = None,
    sobel_y: torch.Tensor = None,
) -> float:
    """Score spatial smoothness via inverse total-variation (Sobel gradient magnitude).

    Implements S_smooth from the paper (Eq. 3).
    """
    if sobel_x is None:
        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            dtype=torch.float32, device=attn_map.device
        ).unsqueeze(0).unsqueeze(0)
    if sobel_y is None:
        sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            dtype=torch.float32, device=attn_map.device
        ).unsqueeze(0).unsqueeze(0)

    inp = attn_map.unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        gx = F.conv2d(inp, sobel_x, padding=1)
        gy = F.conv2d(inp, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2).mean()
    return float(1.0 / (1.0 + mag.item()))


def calculate_layer_position_score(
    layer_idx: int,
    total_layers: int = None,
    cfg: ADAPTConfig = None,
) -> float:
    """Score based on layer depth — favors middle-deep layers (~layer 20 in 32-layer models)."""
    cfg = cfg or default_config
    if total_layers is None:
        total_layers = cfg.TOTAL_ATTENTION_LAYERS

    norm = float(layer_idx) / (total_layers - 1 + 1e-6)
    target = 20.0 / (total_layers - 1 + 1e-6)  # ~0.645 for 32 layers
    score = -4.0 * (norm - target) ** 2 + 1.0
    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# Layer fusion (Section 4.1.1 of paper)
# ---------------------------------------------------------------------------

def calculate_attention_layer_weights(
    attention_layers_data: Dict[int, torch.Tensor],
    total_layers: int = None,
    weight_boundary: float = None,
    weight_frequency: float = None,
    weight_smoothness: float = None,
    weight_concentration: float = None,
    weight_layer_position: float = None,
    cfg: ADAPTConfig = None,
) -> Dict[int, float]:
    """Compute per-layer fusion weights from multi-criteria quality scores.

    Returns a dict mapping layer_idx → normalized weight ∈ [0, 1].
    """
    cfg = cfg or default_config
    if total_layers is None:
        total_layers = cfg.TOTAL_ATTENTION_LAYERS
    if weight_boundary is None:
        weight_boundary = cfg.WEIGHT_BOUNDARY_PENALTY
    if weight_frequency is None:
        weight_frequency = cfg.WEIGHT_FREQUENCY_MATCH
    if weight_smoothness is None:
        weight_smoothness = cfg.WEIGHT_SMOOTHNESS
    if weight_concentration is None:
        weight_concentration = cfg.WEIGHT_CONCENTRATION
    if weight_layer_position is None:
        weight_layer_position = cfg.WEIGHT_LAYER_POSITION

    if not attention_layers_data:
        return {}

    # Pre-create Sobel kernels on the right device
    first_map = next(iter(attention_layers_data.values()))
    dev = first_map.device
    sobel_x = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        dtype=torch.float32, device=dev
    ).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        dtype=torch.float32, device=dev
    ).unsqueeze(0).unsqueeze(0)

    raw_scores: Dict[int, float] = {}
    all_scores = []

    for layer_idx, attn_map in attention_layers_data.items():
        if attn_map.numel() == 0 or attn_map.max() == attn_map.min():
            raw_scores[layer_idx] = 0.0
            all_scores.append(0.0)
            continue

        combined = (
            calculate_boundary_penalty_score(attn_map, cfg=cfg) * weight_boundary
            + calculate_frequency_match_score(attn_map, layer_idx, total_layers, cfg=cfg) * weight_frequency
            + calculate_smoothness_score(attn_map, sobel_x, sobel_y) * weight_smoothness
            + calculate_concentration_score(attn_map) * weight_concentration
            + calculate_layer_position_score(layer_idx, total_layers, cfg=cfg) * weight_layer_position
        )
        raw_scores[layer_idx] = combined
        all_scores.append(combined)

    if not all_scores:
        return {}

    # Min-max normalize → [0, 1]
    mn, mx = min(all_scores), max(all_scores)
    if mx == mn:
        for k in raw_scores:
            raw_scores[k] = 1.0 if raw_scores[k] > 0 else 0.0
    else:
        for k in raw_scores:
            raw_scores[k] = (raw_scores[k] - mn) / (mx - mn)

    # Softmax with temperature scaling
    vals = torch.tensor(list(raw_scores.values()), dtype=torch.float32)
    if vals.numel() > 0 and vals.max() > 0:
        vals = torch.exp(vals * cfg.FREQ_MATCH_SCALING_FACTOR)
        vals = vals / vals.sum()
        for i, k in enumerate(raw_scores):
            raw_scores[k] = vals[i].item()
    elif vals.numel() == 1:
        raw_scores[list(raw_scores.keys())[0]] = 1.0 if vals[0] > 0 else 0.0

    return raw_scores
