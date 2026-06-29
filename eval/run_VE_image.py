"""Convert a regular image into a Visual Enhanced (VE) image.

VE image = original image + cross-attention anchor heatmap overlay.
High-attention regions remain clear (close to the original image),
while low-attention regions fade toward white, making the model's
visual grounding instantly visible for any given question.

Principle (Paper Section 4.1):
  1. Use cross-attention from the first K early tokens to build a visual anchor
  2. Score each layer's attention map on multiple quality dimensions → Softmax fusion
  3. Overlay the fused anchor onto the original image

Usage:
    python eval/run_VE_image.py \
        --model-path liuhaotian/llava-v1.5-7b \
        --image path/to/input.jpg \
        --question "Describe this image in detail." \
        --output-ve ve_output.jpg \
        --gpu 0
"""

import argparse
import os
import sys
from pathlib import Path


def build_ve_image(
    model_path: str,
    image_path: str,
    question: str,
    output_ve: str,
    gpu_id: int = 0,
    mode: str = "quality",
    initial_phase_calls: int = 5,
    temperature: float = 0.2,
    max_new_tokens: int = 256,
    save_anchor_raw: str = None,
    show_top_layers: int = 5,
    model_base: str = None,
    seed: int = 42,
    boundary_penalty_strength: float = 2.0,
    boundary_penalty_threshold: float = 0.6,
    edge_halving_factor: float = 0.5,
):
    """Main pipeline: load model → register hooks → generate → extract anchor → produce VE image.

    Args:
        model_path: Path to LLaVA model or HuggingFace ID.
        image_path: Path to the input image.
        question: Question about the image (different questions yield different VE images).
        output_ve: Path to save the output VE image.
        gpu_id: GPU device index to use (default: 0).
        mode: Attention modification mode, 'quality' (recommended) or 'hhi'.
        initial_phase_calls: Number of early tokens K used to build the anchor.
        temperature: Generation temperature.
        max_new_tokens: Maximum number of new tokens to generate.
        save_anchor_raw: If set, save the raw anchor heatmap to this path.
        show_top_layers: Print the top-N layers ranked by quality weight.
        model_base: LLaVA model_base parameter.
        seed: Random seed.
        boundary_penalty_strength: Edge penalty strength (lower = less penalty).
        boundary_penalty_threshold: Ratio threshold for triggering edge penalty (higher = harder to trigger).
        edge_halving_factor: Edge region attenuation factor (higher = preserves more edge content).
    """
    # Lazy imports (ensure CUDA_VISIBLE_DEVICES is already set in main())
    import torch
    import numpy as np
    from PIL import Image
    from transformers import set_seed

    from adapt.config import ADAPTConfig
    from adapt.hook_logger import hook_logger_multi
    from adapt.anchor import compute_anchor, blend_mask, revise_mask
    from adapt.scorer import calculate_attention_layer_weights

    set_seed(seed)

    # ---- Step 0: Confirm GPU setup ----
    if torch.cuda.is_available():
        print(f"[GPU] CUDA available, device count: {torch.cuda.device_count()}")
        print(f"[GPU] Current device: {torch.cuda.get_device_name(0)}")
    else:
        print("[GPU] Warning: CUDA not available, using CPU")

    # ---- Step 1: Create config ----
    cfg = ADAPTConfig()
    cfg.ANCHOR_NUM_TOKENS = initial_phase_calls
    # Apply edge penalty parameters
    cfg.BOUNDARY_PENALTY_STRENGTH = boundary_penalty_strength
    cfg.BOUNDARY_PENALTY_THRESHOLD_RATIO = boundary_penalty_threshold
    cfg.EDGE_HALVING_FACTOR = edge_halving_factor

    print(f"\n{'='*60}")
    print(f"ADAPT: Visual Enhanced (VE) Image Generation")
    print(f"{'='*60}")
    print(f"  Input image:      {image_path}")
    print(f"  Question:         {question}")
    print(f"  Mode:             {mode}")
    print(f"  Anchor K:         {initial_phase_calls}")
    print(f"  Boundary penalty: {boundary_penalty_strength}")
    print(f"  Boundary thresh:  {boundary_penalty_threshold}")
    print(f"  Edge halving:     {edge_halving_factor}")
    print(f"{'='*60}\n")

    # ---- Step 2: Load LLaVA model ----
    print("[1/5] Loading LLaVA model...")
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    # Ensure the entire model (including vision tower) is on GPU
    model = model.to(device=model.device, dtype=model.dtype)
    print(f"  Model loaded: {model_name}")

    # ---- Step 3: Register ADAPT hooks ----
    print("[2/5] Registering ADAPT attention hooks...")
    hook_loggers = hook_logger_multi(
        model,
        model.device,
        layer_indices=list(range(32)),
        modify_attention=True,
        mode=mode,
        initial_phase_calls=initial_phase_calls,
        adaptive_k1=0.4,
        cfg=cfg,
    )
    print(f"  Registered on {len(hook_loggers)} layers")

    # ---- Step 4: Prepare inputs ----
    print("[3/5] Preparing inputs...")
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    qs = DEFAULT_IMAGE_TOKEN + '\n' + question
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)

    # ---- Step 5: Generate (hooks auto-build anchor + supervise attention) ----
    print("[4/5] Generating (ADAPT hooks working automatically)...")
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().to(model.device),
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    # Decode output
    input_len = input_ids.shape[1]
    response = tokenizer.decode(
        output_ids[0, input_len:], skip_special_tokens=True
    ).strip()

    print(f"\n{'─'*60}")
    print(f"Model response:")
    print(f"{'─'*60}")
    print(response)
    print(f"{'─'*60}\n")

    # ---- Step 6: Extract anchor + generate VE image ----
    print("[5/5] Extracting cross-attention anchor → generating VE image...")

    layer_maps = {}
    for layer_idx, logger in hook_loggers.items():
        anchor_2d = logger.get_anchor_map(avg_tokens=initial_phase_calls)
        if anchor_2d is not None:
            layer_maps[layer_idx] = anchor_2d

    if not layer_maps:
        print("[Error] Failed to extract anchor data from any layer!")
        print("  Possible causes: insufficient generated tokens, or model architecture mismatch.")
        return None

    # Compute layer quality weights + fuse anchor
    weights = calculate_attention_layer_weights(layer_maps, cfg=cfg)
    fused_anchor = compute_anchor(layer_maps, cfg=cfg)

    # Print top layers by quality weight
    sorted_layers = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Layer quality ranking (Top-{show_top_layers}):")
    for rank, (layer_idx, w) in enumerate(sorted_layers[:show_top_layers], 1):
        print(f"    {rank}. Layer {layer_idx}: weight={w:.4f}")

    # ---- Save VE image ----
    print(f"\n  Saving VE image → {output_ve}")
    ve_image = blend_mask(
        image,
        fused_anchor.float(),
        output_path=output_ve,
        cfg=cfg,
    )

    # ---- Optional: save raw anchor heatmap ----
    if save_anchor_raw:
        refined = revise_mask(fused_anchor.float(), cfg=cfg)
        refined_np = (refined.detach().cpu().numpy() * 255).astype(np.uint8)
        anchor_img = Image.fromarray(refined_np, mode="L")
        anchor_img.save(save_anchor_raw)
        print(f"  Raw anchor heatmap → {save_anchor_raw}")

    if ve_image is not None:
        print(f"\nVE image saved to: {output_ve}")
        print(f"  Size: {ve_image.size}")
        print(f"  Formula: VE = image × attention_mask + white × (1 - attention_mask)")
        print(f"  High-attention → clear, low-attention → faded\n")
    else:
        print("[Error] VE image generation failed!")

    return ve_image


def main():
    parser = argparse.ArgumentParser(
        description="ADAPT: Convert a regular image to a Visual Enhanced (VE) image"
    )

    # ---- Required arguments ----
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b",
                        help="Path to LLaVA model or HuggingFace ID")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--question", type=str, default="Describe this image in detail.",
                        help="Question about the image (different questions produce different VE images)")
    parser.add_argument("--output-ve", type=str, default="./ve_output.jpg",
                        help="Path to save the output VE image")

    # ---- GPU ----
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index to use (default: 0)")

    # ---- ADAPT parameters ----
    parser.add_argument("--mode", type=str, default="quality",
                        choices=["quality", "hhi"],
                        help="Attention modification mode: quality (recommended) or hhi")
    parser.add_argument("--initial-phase-calls", type=int, default=5,
                        help="Number of early tokens K for building the anchor (default: 5)")
    parser.add_argument("--model-base", type=str, default=None,
                        help="LLaVA model_base parameter")

    # ---- Generation parameters ----
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Generation temperature (default: 0.2)")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum new tokens to generate (default: 256)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # ---- Edge penalty parameters (control how strongly image borders are suppressed) ----
    parser.add_argument("--boundary-penalty-strength", type=float, default=2.0,
                        help="Edge penalty strength, lower = sharper edges (default: 2.0, range: 0.0~3.0)")
    parser.add_argument("--boundary-penalty-threshold", type=float, default=0.6,
                        help="Ratio threshold to trigger edge penalty, higher = less trigger (default: 0.6, range: 0.3~1.0)")
    parser.add_argument("--edge-halving-factor", type=float, default=0.5,
                        help="Edge region attenuation factor, higher = preserve more edges (default: 0.5, range: 0.1~1.0)")

    # ---- Output options ----
    parser.add_argument("--save-anchor-raw", type=str, default=None,
                        help="Additionally save the raw anchor heatmap to this path")
    parser.add_argument("--show-top-layers", type=int, default=5,
                        help="Number of top-quality layers to print (default: 5)")

    args = parser.parse_args()

    # ---- Must set CUDA_VISIBLE_DEVICES before importing torch ----
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Check input file exists
    if not os.path.exists(args.image):
        print(f"[Error] Input image not found: {args.image}")
        sys.exit(1)

    build_ve_image(
        model_path=args.model_path,
        image_path=args.image,
        question=args.question,
        output_ve=args.output_ve,
        gpu_id=args.gpu,
        mode=args.mode,
        initial_phase_calls=args.initial_phase_calls,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        save_anchor_raw=args.save_anchor_raw,
        show_top_layers=args.show_top_layers,
        model_base=args.model_base,
        seed=args.seed,
        boundary_penalty_strength=args.boundary_penalty_strength,
        boundary_penalty_threshold=args.boundary_penalty_threshold,
        edge_halving_factor=args.edge_halving_factor,
    )


if __name__ == "__main__":
    main()
