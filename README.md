# ADAPT: Attention Dynamics Alignment with Preference Tuning

**ADAPT** is an attention-based framework for reducing hallucinations in Multimodal Large Language Models (MLLMs). It intervenes directly on text-to-image cross-attention dynamics during generation.

## Overview

ADAPT consists of three synergistic stages:

1. **Cross-Attention Visual Anchor** — Extract early, reliable cross-attention, refine it with multi-criteria scoring (spectral consistency, spatial smoothness, adaptive focus), and debias it to produce a stable spatial grounding reference.

2. **Attention-Supervised Inference (ASI)** — Monitor cross-attention against the anchor during decoding. When attention drifts (becomes unfocused or spatially biased), apply sparse corrective steering to re-anchor it to visually grounded regions.

3. **Visual Attention Guidance DPO (VAG-DPO)** — Preference-tune the model using anchor-enhanced images (chosen) vs noise images (rejected) to reinforce visually grounded generation.

## Installation

```bash
pip install -e .
```

Requires a working LLaVA environment (model, tokenizer, image processor).

## Quick Start

```python
from adapt import ADAPTConfig, hook_logger_multi
from adapt.anchor import compute_anchor, refine_cross_attention_anchor

# Load your LLaVA model
# model = ...

# 1. Register ADAPT hooks on attention layers
cfg = ADAPTConfig()
hook_loggers = hook_logger_multi(
    model, model.device,
    layer_indices=list(range(32)),
    modify_attention=True,  # enable attention supervision
    initial_phase_calls=5,  # K: early tokens for anchor
    adaptive_k1=0.4,        # intervention strength
    cfg=cfg,
)

# 2. Generate as usual — hooks apply attention supervision automatically
# output = model.generate(...)

# 3. Extract the cross-attention visual anchor
layer_maps = {}
for idx, logger in hook_loggers.items():
    anchor = logger.get_anchor_map(avg_tokens=5)
    if anchor is not None:
        layer_maps[idx] = anchor

fused_anchor = compute_anchor(layer_maps, cfg=cfg)

# 4. Visualize: overlay anchor on input image
refine_cross_attention_anchor(layer_maps, "input.jpg",
                              output_path="anchor_heatmap.jpg")
```

## AMBER Benchmark Evaluation

```bash
python -m eval.run_amber \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder /path/to/AMBER/images \
    --question-file /path/to/AMBER/questions.json \
    --answers-file ./results/amber_results.jsonl \
    --use-adapt \
    --adaptive-k1 0.4
```

## Demo

```bash
python examples/demo.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --image path/to/image.jpg \
    --question "Describe this image in detail." \
    --save-anchor anchor_output.jpg
```

## Core Modules

| Module | Description |
|--------|-------------|
| `adapt.config` | All configuration constants (`ADAPTConfig`) |
| `adapt.hook_manager` | Lightweight PyTorch hook registration system |
| `adapt.hook_logger` | `MaskHookLogger` — records/modifies cross-attention per layer |
| `adapt.scorer` | Multi-criteria attention quality scoring functions |
| `adapt.anchor` | Anchor refinement pipeline (normalize → enhance → smooth → debias → blend) |
| `adapt.inference` | AMC scoring and attention steering operator (Section 4.2) |
| `eval.run_amber` | AMBER benchmark evaluation with ADAPT enabled |

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_phase_calls` (K) | 5 | Number of early tokens for building the anchor |
| `adaptive_k1` | 0.4 | Intervention strength in HHI-based mode |
| `AMC_THRESHOLD_TAU` | 0.6 | AMC threshold for triggering steering |
| `STEERING_STRENGTH_ALPHA` | 0.5 | Log-prior injection strength |

## Citation

If you use ADAPT in your research, please cite:

```
@article{yao2025adapt,
  title={ADAPT: Attention Dynamics Alignment with Preference Tuning
         for Faithful MLLMs},
  author={Yao, Zhiyuan and Fu, Zheren and Zheng, Zhixiao and
          Li, Jiajun and Tu, Yi and Mao, Zhendong},
  year={2025}
}
```

## License

This project is released under the MIT License.
