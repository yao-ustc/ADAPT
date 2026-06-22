"""Configuration constants for ADAPT framework."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ADAPTConfig:
    """Central configuration for ADAPT attention-based hallucination mitigation."""

    # ---- Image / Patch dimensions ----
    IMAGE_PATCH_DIM: int = 24          # Attention map spatial dim (24x24 patches)
    TOTAL_ATTENTION_LAYERS: int = 32   # LLaVA-v1.5 has 32 layers

    # ---- Token slicing (LLaVA-v1.5 7B) ----
    ATTENTION_SLICE_START: int = 35    # First image token index
    ATTENTION_SLICE_END: int = 611     # Last image token index + 1
    NUM_IMAGE_TOKENS: int = 576        # 24 * 24

    # ---- Visual Anchor: early tokens ----
    ANCHOR_NUM_TOKENS: int = 5         # K: number of early tokens for anchor

    # ---- Layer fusion weights (Section 4.1, paper) ----
    WEIGHT_BOUNDARY_PENALTY: float = 0.25
    WEIGHT_FREQUENCY_MATCH: float = 0.30
    WEIGHT_SMOOTHNESS: float = 0.15
    WEIGHT_CONCENTRATION: float = 0.20
    WEIGHT_LAYER_POSITION: float = 0.10

    # ---- Fusion weight projection (omega in paper) ----
    OMEGA_SPEC: float = 0.4    # Spectral consistency weight
    OMEGA_SMOOTH: float = 0.3  # Spatial smoothness weight
    OMEGA_FOCUS: float = 0.3   # Adaptive focus weight

    # ---- Scoring parameters ----
    BOUNDARY_PATCH_THICKNESS: int = 2
    BOUNDARY_PENALTY_THRESHOLD_RATIO: float = 0.6
    BOUNDARY_PENALTY_STRENGTH: float = 2.0

    FFT_RADIAL_CUTOFF_RATIO: float = 0.25
    FREQ_MATCH_SCALING_FACTOR: float = 10.0

    # ---- Anchor refinement ----
    DEFAULT_ENHANCE_COE: float = 10.0
    DEFAULT_KERNEL_SIZE: int = 3
    EDGE_HALVING_FACTOR: float = 0.5
    POINT_SMOOTHING_KERNEL_SIZE: int = 5
    HEATMAP_THRESHOLD_PERCENTILE: float = 0.4

    # ---- Attention-Supervised Inference (Section 4.2, paper) ----
    AMC_THRESHOLD_TAU: float = 0.6    # τ: trigger threshold for steering
    STEERING_STRENGTH_ALPHA: float = 0.5  # α: log-prior injection strength

    # ---- HHI-based fallback mode ----
    HHI_INITIAL_PHASE_CALLS: int = 5
    HHI_ADAPTIVE_K1: float = 0.4

    # ---- Layer indices for hook registration ----
    DEFAULT_LAYER_INDICES: List[int] = field(default_factory=lambda: list(range(32)))

    # ---- DPO-related ----
    DPO_BETA: float = 0.1


# Singleton-like default config
default_config = ADAPTConfig()
