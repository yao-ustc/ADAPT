"""Quick demo of ADAPT attention supervision.

1. Load a LLaVA model
2. Register ADAPT hooks
3. Generate a response with attention-supervised inference
4. Show the cross-attention anchor

Usage:
    python examples/demo.py \
        --model-path liuhaotian/llava-v1.5-7b \
        --image path/to/image.jpg \
        --question "What is in the image?"
"""

import argparse
import os
import sys
import torch
from PIL import Image
from pathlib import Path

# Make sure ADAPT is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapt.config import ADAPTConfig
from adapt.hook_logger import hook_logger_multi
from adapt.anchor import compute_anchor, refine_cross_attention_anchor
from adapt.scorer import calculate_attention_layer_weights


def demo():
    parser = argparse.ArgumentParser(description="ADAPT quick demo")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, default="Describe this image in detail.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--mode", type=str, default="quality",
                        choices=["hhi", "quality"],
                        help="Attention modification mode: hhi or quality")
    parser.add_argument("--use-adapt", action="store_true", default=True)
    parser.add_argument("--save-anchor", type=str, default=None,
                        help="Path to save the anchor heatmap image.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import set_seed
    set_seed(args.seed)

    # --- 1. Load model ---
    print("[1/5] Loading LLaVA model...")
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name
    )
    print(f"  Loaded: {model_name}")

    # --- 2. Setup ADAPT hooks ---
    cfg = ADAPTConfig()
    print("[2/5] Registering ADAPT hooks...")
    hl = hook_logger_multi(
        model, model.device,
        layer_indices=list(range(32)),
        modify_attention=args.use_adapt,
        mode=args.mode,
        initial_phase_calls=5,
        adaptive_k1=0.4,
        cfg=cfg,
    )
    print(f"  Registered on {len(hl)} layers (modify={args.use_adapt})")

    # --- 3. Prepare inputs ---
    print("[3/5] Preparing inputs...")
    image = Image.open(args.image).convert("RGB")
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    qs = DEFAULT_IMAGE_TOKEN + '\n' + args.question
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)

    # --- 4. Generate ---
    print("[4/5] Generating with ADAPT...")
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().to(model.device),
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    input_len = input_ids.shape[1]
    response = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()
    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"{'='*60}")
    print(f"Response: {response}")
    print(f"{'='*60}")

    # --- 5. Extract anchor (optional) ---
    print("[5/5] Extracting cross-attention anchor...")

    # Collect attention from early tokens across layers
    layer_maps = {}
    for layer_idx, logger in hl.items():
        anchor_2d = logger.get_anchor_map(avg_tokens=5)
        if anchor_2d is not None:
            layer_maps[layer_idx] = anchor_2d

    if layer_maps:
        # Compute fused anchor
        fused_anchor = compute_anchor(layer_maps, cfg=cfg)
        weights = calculate_attention_layer_weights(layer_maps, cfg=cfg)

        # Show top-5 layers by weight
        sorted_layers = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  Top-5 layers by quality weight: "
              f"{', '.join(f'L{idx}={w:.3f}' for idx, w in sorted_layers)}")

        # Save anchor heatmap
        if args.save_anchor:
            refine_cross_attention_anchor(
                layer_maps, image,
                output_path=args.save_anchor,
                cfg=cfg,
            )
            print(f"  Anchor heatmap saved to: {args.save_anchor}")
    else:
        print("  Warning: No anchor data collected.")

    print("\nDone.")


if __name__ == "__main__":
    demo()
