"""Attention-Supervised Inference (paper Section 4.2).

Implements the Anchor-Modulated Concentration (AMC) scoring and the
sparse attention steering operator that corrects cross-attention drift
during autoregressive decoding.

Key equations from the paper:
  S_AMC(A_t || A_anchor) = Σ(a_i² · w_i^anchor) / (Σ a_i)²       (Eq. 6)
  â_i = a_i + α · log(w_i^anchor + ε)   if S_AMC < τ              (Eq. 7)
"""

import torch
from typing import Optional, Dict
from .config import ADAPTConfig, default_config


def compute_amc_score(
    attn_weights: torch.Tensor,
    anchor_weights: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute the Anchor-Modulated Concentration (AMC) fidelity score.

    AMC combines concentration (a_i² term) with anchor consistency
    (down-weighting regions where anchor is near-zero), making it suitable
    for online drift detection during autoregressive decoding.

    Args:
        attn_weights: current cross-attention distribution, shape [N] or [B, N].
        anchor_weights: anchor weights, same shape as attn_weights.
        eps: numerical stability.

    Returns:
        S_AMC ∈ [0, 1]. Higher = better alignment with anchor.
    """
    # Ensure 1D
    a = attn_weights.flatten().float()
    w = anchor_weights.flatten().float()

    numerator = (a ** 2 * w).sum()
    denominator = (a.sum() ** 2) + eps

    return numerator / denominator


def steer_attention(
    attn_logits: torch.Tensor,
    anchor_weights: torch.Tensor,
    alpha: float = None,
    tau: float = None,
    eps: float = 1e-9,
    cfg: ADAPTConfig = None,
) -> torch.Tensor:
    """Apply sparse corrective steering to attention logits.

    Only activates when S_AMC < τ (attention has drifted).
    Adds an anchor-derived log-prior to gently bias attention back toward
    visually grounded regions.

    Args:
        attn_logits: pre-softmax attention logits for image tokens, shape [N].
        anchor_weights: anchor weights [N] (non-negative, sums to ~1).
        alpha: steering strength (paper α).
        tau: AMC threshold (paper τ = 0.6).
        eps: numerical stability for log.

    Returns:
        Modified attention logits [N].
    """
    cfg = cfg or default_config
    if alpha is None:
        alpha = cfg.STEERING_STRENGTH_ALPHA
    if tau is None:
        tau = cfg.AMC_THRESHOLD_TAU

    # First compute softmax to check current attention
    attn = torch.softmax(attn_logits.float(), dim=-1)
    score = compute_amc_score(attn, anchor_weights, eps)

    if score >= tau:
        return attn_logits  # No intervention needed

    # Apply steering: â_i = a_i + α · log(w_i + ε)
    log_prior = torch.log(anchor_weights.float() + eps)
    steered = attn_logits.float() + alpha * log_prior.to(attn_logits.device)
    return steered.to(attn_logits.dtype)


class AttentionSupervisedInference:
    """High-level controller for attention-supervised decoding.

    Manages the anchor reference and applies AMC-based steering across
    multiple layers during generation.

    Usage:
        asi = AttentionSupervisedInference(model, cfg)
        asi.build_anchor_from_hooks(hook_loggers)
        # ... during generation, hooks call asi.step() each token ...
    """

    def __init__(self, cfg: ADAPTConfig = None):
        self.cfg = cfg or default_config
        self.anchor: Optional[torch.Tensor] = None       # [N] flattened
        self.anchor_2d: Optional[torch.Tensor] = None    # [24, 24]
        self.intervention_count: int = 0
        self.total_steps: int = 0

    def build_anchor_from_loggers(
        self,
        hook_loggers: Dict[int, "MaskHookLogger"],
    ):
        """Extract and fuse anchor from early-phase hook loggers.

        Averages attention across layers and the first K tokens.
        """
        anchors_2d = []
        for logger in hook_loggers.values():
            anchor_2d = logger.get_anchor_map()
            if anchor_2d is not None:
                anchors_2d.append(anchor_2d)

        if not anchors_2d:
            return

        # Average across layers
        stacked = torch.stack(anchors_2d, dim=0)  # [L, 24, 24]
        self.anchor_2d = stacked.mean(dim=0)       # [24, 24]
        self.anchor = self.anchor_2d.flatten()     # [576]

    def build_anchor_from_tensor(
        self,
        anchor_2d: torch.Tensor,
    ):
        """Set anchor directly from a pre-computed [24, 24] tensor."""
        self.anchor_2d = anchor_2d
        self.anchor = anchor_2d.flatten()

    def step(
        self,
        attn_logits: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """Apply one step of attention supervision.

        Args:
            attn_logits: current attention logits [N_heads, N_image_tokens].
            layer_idx: which layer (for diagnostics).

        Returns:
            Possibly steered attention logits.
        """
        if self.anchor is None:
            return attn_logits

        self.total_steps += 1
        result = steer_attention(attn_logits, self.anchor, cfg=self.cfg)

        if not torch.equal(result, attn_logits):
            self.intervention_count += 1

        return result

    @property
    def intervention_rate(self) -> float:
        """Fraction of steps where steering was applied."""
        if self.total_steps == 0:
            return 0.0
        return self.intervention_count / self.total_steps
