"""MaskHookLogger — the core attention hook for recording and modifying
text-to-image cross-attention during autoregressive decoding.

Supports two operating modes:
  1. **HHI-based** (default, ``mode='hhi'``): adaptive mixing based on concentration drift.
     Detects when attention becomes more "diffuse" than the early-phase baseline
     and blends in the stored anchor proportionally.

  2. **Quality-based redistribution** (``mode='quality'``): evaluates attention quality
     per step via normalize→enhance→smooth→threshold, zeros out low-quality regions,
     and redistributes that mass to salient regions. This is the primary mode from
     hook_714.py corresponding to the paper's Attention-Supervised Inference (Section 4.2).
"""

import torch
import torch.nn.functional as F
import warnings
from typing import Dict, List, Optional

from .config import ADAPTConfig, default_config
from .hook_manager import HookManager, init_hookmanager


class MaskHookLogger:
    """Record and optionally modify text-to-image cross-attention at a specific layer.

    Hooks into the 'after_softmax' point of a self-attention module. During
    autoregressive generation (query_len == 1), it:
      - Extracts the image-token slice [st:ed].
      - In 'hhi' mode: accumulates early attention → baseline HHI, then detects
        dispersion and blends in the stored anchor.
      - In 'quality' mode: evaluates attention quality per-step, identifies and
        zeros out low-quality patches, then redistributes that mass to salient regions.

    Parameters
    ----------
    model : nn.Module
        The parent MLLM (used for device/dtype reference).
    device : torch.device
    layer_idx : int, optional
        Which Transformer layer this logger is attached to.
    mode : str
        ``'hhi'`` for HHI-based blending, ``'quality'`` for quality-based redistribution.
    initial_phase_calls : int
        Number of early tokens used to build the anchor (K in paper).
    adaptive_k1 : float
        Intervention strength for HHI mode.
    cfg : ADAPTConfig
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        device: torch.device,
        layer_idx: int = -1,
        mode: str = "hhi",
        initial_phase_calls: int = None,
        adaptive_k1: float = None,
        cfg: ADAPTConfig = None,
    ):
        self.cfg = cfg or default_config
        self.device = device
        self.model = model
        self._layer_idx = layer_idx
        self.mode = mode

        # Slice range for image tokens (LLaVA-v1.5 7B default: 35→611)
        self.st = self.cfg.ATTENTION_SLICE_START
        self.ed = self.cfg.ATTENTION_SLICE_END
        self.patch_dim = self.cfg.IMAGE_PATCH_DIM

        # Phase tracking
        self.initial_phase_calls = initial_phase_calls or self.cfg.HHI_INITIAL_PHASE_CALLS
        self.adaptive_k1 = adaptive_k1 or self.cfg.HHI_ADAPTIVE_K1
        self.call_count = 0

        # Anchor / baseline storage (HHI mode)
        self.accumulated: List[torch.Tensor] = []
        self.stored_avg_attn: Optional[torch.Tensor] = None
        self.benchmark_hhi: Optional[float] = None

        # Recorded attention tensors
        self.attns: List[torch.Tensor] = []

        # ---- Pre-allocated components for quality mode ----
        # Determine dtype from model
        try:
            self._model_dtype = model.config.torch_dtype
            if self._model_dtype == torch.float16:
                self._model_dtype = torch.float16
            else:
                self._model_dtype = torch.float32
        except Exception:
            self._model_dtype = torch.float32

        # 1. Sobel operators for smoothness scoring
        self.sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
            dtype=self._model_dtype, device=self.device
        ).unsqueeze(0).unsqueeze(0)
        self.sobel_y = torch.tensor(
            [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
            dtype=self._model_dtype, device=self.device
        ).unsqueeze(0).unsqueeze(0)

        # 2. Smoothing conv (kernel_size=1 → identity, effectively disabled by default)
        ks_smooth = self.cfg.DEFAULT_KERNEL_SIZE
        pad_smooth = (ks_smooth - 1) // 2
        self.conv_smooth = torch.nn.Conv2d(
            1, 1, kernel_size=ks_smooth, padding=pad_smooth,
            padding_mode="replicate", stride=1, bias=False,
            dtype=self._model_dtype,
        )
        self.conv_smooth.weight.data = (
            torch.ones_like(self.conv_smooth.weight.data) / (ks_smooth ** 2)
        )
        self.conv_smooth.to(self.device)

        # 3. Point-smoothing conv (larger kernel to diffuse isolated peaks)
        ks_point = self.cfg.POINT_SMOOTHING_KERNEL_SIZE
        if ks_point > 1 and ks_point % 2 == 1:
            pad_point = (ks_point - 1) // 2
            self.conv_point_smooth = torch.nn.Conv2d(
                1, 1, kernel_size=ks_point, padding=pad_point,
                padding_mode="replicate", stride=1, bias=False,
                dtype=self._model_dtype,
            )
            self.conv_point_smooth.weight.data = (
                torch.ones_like(self.conv_point_smooth.weight.data) / (ks_point ** 2)
            )
            self.conv_point_smooth.to(self.device)
        else:
            self.conv_point_smooth = None

    @property
    def layer_idx(self) -> int:
        return self._layer_idx

    # ------------------------------------------------------------------
    # Utility helpers (quality mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(mat: torch.Tensor, method: str = "min") -> torch.Tensor:
        if mat.numel() == 0 or mat.max() == mat.min():
            return torch.ones_like(mat) * 0.5
        if method == "max":
            return (mat.max() - mat) / (mat.max() - mat.min())
        elif method == "min":
            return (mat - mat.min()) / (mat.max() - mat.min())
        raise NotImplementedError(f"Unknown method: {method}")

    @staticmethod
    def _enhance(mat: torch.Tensor, coe: float = 10.0) -> torch.Tensor:
        if mat.std() == 0:
            return torch.ones_like(mat) * 0.5
        mat = (mat - mat.mean()) / mat.std()
        mat = mat * coe
        return torch.sigmoid(mat).clamp(0, 1)

    def _revise_mask_internal(self, mask: torch.Tensor) -> torch.Tensor:
        """Normalize → enhance → smooth (quality-mode internal pipeline)."""
        mask = self._normalize(mask, "min")
        mask = self._enhance(mask, coe=self.cfg.DEFAULT_ENHANCE_COE)
        inp = mask.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            mask = self.conv_smooth(inp)
        return mask.squeeze(0).squeeze(0)

    # ------------------------------------------------------------------
    # Quality scoring (quality mode internals)
    # ------------------------------------------------------------------

    def _calc_boundary_penalty(self, attn_map: torch.Tensor) -> float:
        H, W = attn_map.shape
        t = self.cfg.BOUNDARY_PATCH_THICKNESS
        if t * 2 >= H or t * 2 >= W:
            return 1.0

        top = attn_map[:t, :]
        bottom = attn_map[-t:, :]
        left = attn_map[:, :t]
        right = attn_map[:, -t:]
        boundary_vals = torch.cat([
            top.flatten(), bottom.flatten(),
            left[:, t:-t].flatten(), right[:, t:-t].flatten(),
        ])
        center_vals = attn_map[t:-t, t:-t]
        eps = 1e-6
        mean_b = boundary_vals.mean()
        mean_c = center_vals.mean()
        if mean_c < eps and mean_b < eps:
            return 1.0
        ratio = mean_b / mean_c if mean_c > eps else (float("inf") if mean_b > eps else 1.0)
        if ratio > self.cfg.BOUNDARY_PENALTY_THRESHOLD_RATIO:
            score = 1.0 - min(1.0, (ratio - self.cfg.BOUNDARY_PENALTY_THRESHOLD_RATIO) * self.cfg.BOUNDARY_PENALTY_STRENGTH)
            return max(0.0, score)
        return 1.0

    def _calc_frequency_match(self, attn_map: torch.Tensor) -> float:
        if attn_map.numel() == 0 or attn_map.max() == attn_map.min():
            return 0.5
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
        cutoff = max_r * self.cfg.FFT_RADIAL_CUTOFF_RATIO
        high_mask = radii > cutoff

        total_energy = power.sum()
        if total_energy < 1e-6:
            return 0.5

        high_ratio = (power * high_mask).sum() / (total_energy + 1e-6)
        total_l = self.cfg.TOTAL_ATTENTION_LAYERS
        norm_layer = self._layer_idx / (total_l - 1 + 1e-6)
        ideal_high = 0.8 * (1.0 - norm_layer) + 0.2 * norm_layer
        return float((1.0 - torch.abs(high_ratio - ideal_high)).item())

    def _calc_concentration(self, attn_map: torch.Tensor) -> float:
        if attn_map.numel() == 0 or attn_map.max() == attn_map.min():
            return 0.0
        norm = attn_map / (attn_map.sum() + 1e-6)
        entropy = -(norm * torch.log(norm + 1e-9)).sum()
        max_entropy = torch.log(torch.tensor(attn_map.numel(), dtype=torch.float32, device=attn_map.device) + 1e-9)
        if max_entropy < 1e-6:
            return 0.0
        return float((1.0 - entropy / max_entropy).clamp(0, 1).item())

    def _calc_layer_position(self) -> float:
        total_l = self.cfg.TOTAL_ATTENTION_LAYERS
        norm = self._layer_idx / (total_l - 1 + 1e-6)
        target = 20.0 / (total_l - 1 + 1e-6)
        score = -4.0 * (norm - target) ** 2 + 1.0
        return max(0.0, min(1.0, score))

    def _calc_quality_score(self, attn_map: torch.Tensor) -> float:
        """Combine 5 criteria → single quality score for this layer's attention."""
        if attn_map.numel() == 0 or attn_map.max() == attn_map.min():
            return 0.0

        # Move to device where kernels live, and use float32
        a = attn_map.float().to(self.device)

        boundary = self._calc_boundary_penalty(a)
        frequency = self._calc_frequency_match(a)

        # Smoothness via Sobel
        inp = a.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            gx = F.conv2d(inp, self.sobel_x.float(), padding=1)
            gy = F.conv2d(inp, self.sobel_y.float(), padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2).mean()
        smoothness = float(1.0 / (1.0 + grad_mag.item()))

        concentration = self._calc_concentration(a)
        layer_pos = self._calc_layer_position()

        cfg = self.cfg
        return float(
            boundary * cfg.WEIGHT_BOUNDARY_PENALTY
            + frequency * cfg.WEIGHT_FREQUENCY_MATCH
            + smoothness * cfg.WEIGHT_SMOOTHNESS
            + concentration * cfg.WEIGHT_CONCENTRATION
            + layer_pos * cfg.WEIGHT_LAYER_POSITION
        )

    # ------------------------------------------------------------------
    # HHI helpers (hhi mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hhi(attn_flat: torch.Tensor, patch_dim: int = 24) -> torch.Tensor:
        B, N = attn_flat.shape
        attn_2d = attn_flat.view(B, patch_dim, patch_dim).clone().detach()
        t = 2
        if t * 2 < patch_dim:
            attn_2d[:, :t, :] *= 0.5
            attn_2d[:, -t:, :] *= 0.5
            attn_2d[:, :, :t] *= 0.5
            attn_2d[:, :, -t:] *= 0.5
        return (attn_2d.reshape(B, N) ** 2).sum(dim=-1)

    # ------------------------------------------------------------------
    # Mode 1: HHI-based adaptive blending
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _modify_attention_hhi(self, ret: torch.Tensor) -> torch.Tensor:
        """HHI-based: blend stored anchor when current attention becomes diffuse."""
        is_q_one = ret.shape[2] == 1
        if not is_q_one or ret.shape[-1] < self.ed:
            return ret

        self.call_count += 1
        attn_slice = ret[:, :, :, self.st : self.ed].clone()

        if self.call_count <= self.initial_phase_calls:
            self.accumulated.append(attn_slice.detach().cpu())
            if self.call_count == self.initial_phase_calls and self.accumulated:
                stacked = torch.stack(self.accumulated, dim=0)
                self.stored_avg_attn = stacked.mean(dim=0).to(self.device)
                self.accumulated.clear()
                avg_flat = self.stored_avg_attn.mean(dim=1, keepdim=False).squeeze(1)
                hhi_batch = self._compute_hhi(avg_flat, self.patch_dim)
                self.benchmark_hhi = hhi_batch.mean().item()
            self.attns.append(attn_slice.cpu())
        else:
            if self.stored_avg_attn is None or self.benchmark_hhi is None:
                self.attns.append(attn_slice.cpu())
                return ret

            if self.stored_avg_attn.device != attn_slice.device:
                self.stored_avg_attn = self.stored_avg_attn.to(attn_slice.device)

            current_flat = attn_slice.mean(dim=1, keepdim=False).squeeze(1)
            hhi_current = self._compute_hhi(current_flat, self.patch_dim)
            dispersion = torch.clamp(
                1.0 - (hhi_current / (self.benchmark_hhi + 1e-9)), 0.0, 1.0
            )
            w = torch.clamp(self.adaptive_k1 * dispersion, 0.0, 1.0)
            w = w.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

            modified = (1.0 - w) * attn_slice + w * self.stored_avg_attn
            ret[:, :, :, self.st : self.ed] = modified
            self.attns.append(modified.cpu())

        return ret

    # ------------------------------------------------------------------
    # Mode 2: Quality-based redistribution (from hook_714.py)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _modify_attention_quality(self, ret: torch.Tensor) -> torch.Tensor:
        """Quality-based: zero-out low-quality regions, redistribute to salient ones.

        Pipeline:
          1. Aggregate multi-head attention → 2D [24, 24]
          2. Refine mask: normalize → enhance → smooth → edge-halve → point-smooth
          3. Threshold at HEATMAP_THRESHOLD_PERCENTILE → identify low-quality regions
          4. Zero-out those regions across all heads
          5. Redistribute zeroed mass to remaining regions proportionally
        """
        is_q_one = ret.shape[2] == 1
        if not is_q_one or ret.shape[-1] < self.ed:
            return ret

        self.call_count += 1
        attn_slice = ret[:, :, :, self.st : self.ed].clone()

        # --- Step 1: aggregate heads → 2D ---
        avg_over_heads = attn_slice.mean(dim=1, keepdim=True)  # [B, 1, 1, P]

        if avg_over_heads.shape[0] != 1:
            warnings.warn(
                f"Layer {self._layer_idx}: batch > 1, using first item only."
            )

        attn_2d = avg_over_heads.squeeze(0).squeeze(0).view(self.patch_dim, self.patch_dim)

        # --- Step 2: refine mask ---
        refined = self._revise_mask_internal(attn_2d.clone())

        # Edge halving
        H, W = refined.shape
        t = self.cfg.BOUNDARY_PATCH_THICKNESS
        if t * 2 < H and t * 2 < W:
            bm = torch.zeros_like(refined, dtype=torch.bool, device=self.device)
            bm[:t, :] = True
            bm[-t:, :] = True
            bm[:, :t] = True
            bm[:, -t:] = True
            refined[bm] *= self.cfg.EDGE_HALVING_FACTOR

        # Point smoothing
        if self.conv_point_smooth is not None:
            inp = refined.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                refined = self.conv_point_smooth(inp).squeeze(0).squeeze(0)

        # Re-normalize before threshold
        refined = self._normalize(refined, "min")

        # --- Step 3: threshold → zero-mask ---
        if refined.numel() > 0 and refined.max() > refined.min():
            thresh = torch.quantile(
                refined.flatten().to(torch.float32),
                self.cfg.HEATMAP_THRESHOLD_PERCENTILE,
            )
            mask_zero = refined < thresh  # [H, W] bool
        else:
            mask_zero = torch.zeros_like(refined, dtype=torch.bool, device=self.device)

        # Expand to full attention shape: [1, H, 1, P]
        flat_mask = mask_zero.flatten()
        expanded = flat_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        expanded = expanded.expand_as(attn_slice)

        # --- Step 4: zero-out & redistribute ---
        zeroed_sum = attn_slice[expanded].sum()
        attn_slice[expanded] = 0.0

        non_zero_mask = ~expanded
        n_nonzero = non_zero_mask.sum()

        if n_nonzero > 0 and zeroed_sum > 0:
            current_nonzero = attn_slice[non_zero_mask]
            sum_nonzero = current_nonzero.sum()
            if sum_nonzero > 0:
                redist = (current_nonzero / sum_nonzero) * zeroed_sum
                attn_slice[non_zero_mask] += redist
            else:
                attn_slice[non_zero_mask] += zeroed_sum / n_nonzero

            attn_slice.clamp_(0.0, 1.0)

        # Write back
        ret[:, :, :, self.st : self.ed] = attn_slice
        self.attns.append(attn_slice.cpu())

        return ret

    # ------------------------------------------------------------------
    # Main entry: dispatches to the selected mode
    # ------------------------------------------------------------------

    @torch.no_grad()
    def modify_attention(self, ret: torch.Tensor) -> torch.Tensor:
        """Dispatch to the active modification mode."""
        if self.mode == "quality":
            return self._modify_attention_quality(ret)
        else:
            return self._modify_attention_hhi(ret)

    @torch.no_grad()
    def save_attention(self, ret: torch.Tensor) -> torch.Tensor:
        """Only record attention — no modification."""
        is_q_one = ret.shape[2] == 1
        if is_q_one and ret.shape[-1] >= self.ed:
            self.attns.append(ret[:, :, :, self.st : self.ed].cpu())
        return ret

    @torch.no_grad()
    def zero_attention(self, ret: torch.Tensor) -> torch.Tensor:
        """Zero out all image-token attention (ablation experiment)."""
        is_q_one = ret.shape[2] == 1
        if is_q_one and ret.shape[-1] >= self.ed:
            zero = torch.zeros_like(ret[:, :, :, self.st : self.ed])
            ret[:, :, :, self.st : self.ed] = zero
            self.attns.append(ret[:, :, :, self.st : self.ed].cpu())
        return ret

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reinit(self):
        """Reset all state for a new sample."""
        self.attns.clear()
        self.call_count = 0
        self.accumulated.clear()
        self.stored_avg_attn = None
        self.benchmark_hhi = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def finalize(self) -> Optional[List[torch.Tensor]]:
        """Return collected attention tensors (list of [1, H, 1, P])."""
        if not self.attns:
            return None
        return self.attns

    def get_anchor_map(self, avg_tokens: int = None) -> Optional[torch.Tensor]:
        """Return the stored anchor as a 2D [24, 24] map.

        Args:
            avg_tokens: number of early tokens to average (default: initial_phase_calls).
        """
        if avg_tokens is None:
            avg_tokens = self.initial_phase_calls
        if not self.attns:
            return None

        n = min(avg_tokens, len(self.attns))
        early = torch.cat(self.attns[:n], dim=0)  # [T, H, 1, P]
        avg = early.mean(dim=[0, 1]).squeeze()     # [P]
        try:
            return avg.view(self.patch_dim, self.patch_dim)
        except RuntimeError:
            return None


# ---------------------------------------------------------------------------
# Multi-layer hook registration
# ---------------------------------------------------------------------------

def hook_logger_multi(
    model: "torch.nn.Module",
    device: torch.device,
    layer_indices: List[int] = None,
    modify_attention: bool = True,
    mode: str = "hhi",
    initial_phase_calls: int = None,
    adaptive_k1: float = None,
    cfg: ADAPTConfig = None,
) -> Dict[int, MaskHookLogger]:
    """Create and register MaskHookLogger instances on multiple Transformer layers.

    This is the main entry point for setting up ADAPT on a LLaVA-style model.

    Args:
        model: the MLLM (e.g. LlavaLlamaForCausalLM).
        device: torch device.
        layer_indices: which layers to hook (default: all 32).
        modify_attention: if True, register modify_attention; else save_attention only.
        mode: ``'hhi'`` for HHI-based blending or ``'quality'`` for quality-based
              redistribution (the paper's primary method).
        initial_phase_calls: K — number of early tokens for anchor.
        adaptive_k1: intervention strength (HHI mode only).

    Returns:
        Dict mapping layer_idx → MaskHookLogger.
    """
    cfg = cfg or default_config
    if layer_indices is None:
        layer_indices = cfg.DEFAULT_LAYER_INDICES
    if initial_phase_calls is None:
        initial_phase_calls = cfg.HHI_INITIAL_PHASE_CALLS
    if adaptive_k1 is None:
        adaptive_k1 = cfg.HHI_ADAPTIVE_K1

    hook_loggers: Dict[int, MaskHookLogger] = {}

    for layer_idx in layer_indices:
        try:
            target = model.model.layers[layer_idx].self_attn
        except AttributeError:
            print(f"[ADAPT] Could not find self_attn at layer {layer_idx}. Skipping.")
            continue

        init_hookmanager(target)

        logger = MaskHookLogger(
            model, device,
            layer_idx=layer_idx,
            mode=mode,
            initial_phase_calls=initial_phase_calls,
            adaptive_k1=adaptive_k1,
            cfg=cfg,
        )

        hook_fn = logger.modify_attention if modify_attention else logger.save_attention
        target.hook_manager.register("after_softmax", hook_fn)
        hook_loggers[layer_idx] = logger

    model.hookloggers = hook_loggers
    return hook_loggers
