"""AMBER benchmark evaluation with ADAPT attention supervision.

AMBER (https://github.com/junyangwang0410/AMBER) is a standard hallucination
benchmark for MLLMs. This script evaluates a LLaVA model with ADAPT's
attention-supervised inference enabled.

Usage:
    python -m eval.run_amber \
        --model-path liuhaotian/llava-v1.5-7b \
        --image-folder /path/to/amber/images \
        --question-file /path/to/amber/questions.json \
        --answers-file ./results/amber_results.jsonl \
        --use-adapt \
        --adaptive-k1 0.4
"""

import argparse
import os
import sys
import json
import torch
import warnings
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from transformers import set_seed

from adapt.config import ADAPTConfig
from adapt.hook_logger import hook_logger_multi

warnings.filterwarnings("ignore")


class AMBEREvaluator:
    """Run AMBER evaluation with ADAPT attention hooks."""

    def __init__(
        self,
        model_path: str,
        model_base: str = None,
        conv_mode: str = "llava_v1",
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = 512,
        seed: int = 42,
        # ADAPT settings
        use_adapt: bool = True,
        modify_attention: bool = True,
        mode: str = "quality",
        initial_phase_calls: int = 5,
        adaptive_k1: float = 0.4,
        layer_indices: list = None,
        cfg: ADAPTConfig = None,
    ):
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.seed = seed

        self.use_adapt = use_adapt
        self.modify_attention = modify_attention
        self.mode = mode
        self.initial_phase_calls = initial_phase_calls
        self.adaptive_k1 = adaptive_k1
        self.layer_indices = layer_indices
        self.cfg = cfg or ADAPTConfig()

        self._model = None
        self._tokenizer = None
        self._image_processor = None
        self._context_len = None
        self._hook_loggers = None

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self):
        """Load LLaVA model, tokenizer, and image processor."""
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init
        from llava.mm_utils import get_model_name_from_path

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self._tokenizer, self._model, self._image_processor, self._context_len = \
            load_pretrained_model(self.model_path, self.model_base, model_name)

        print(f"[ADAPT] Loaded model: {model_name}")

    def setup_hooks(self):
        """Register ADAPT attention hooks on the model."""
        if not self.use_adapt:
            return

        layer_indices = self.layer_indices or list(range(32))
        self._hook_loggers = hook_logger_multi(
            self._model,
            self._model.device,
            layer_indices=layer_indices,
            modify_attention=self.modify_attention,
            mode=self.mode,
            initial_phase_calls=self.initial_phase_calls,
            adaptive_k1=self.adaptive_k1,
            cfg=self.cfg,
        )
        print(f"[ADAPT] Registered hooks on {len(self._hook_loggers)} layers "
              f"(mode={self.mode}, modify={self.modify_attention}, K={self.initial_phase_calls})")

    # ------------------------------------------------------------------
    # Single sample evaluation
    # ------------------------------------------------------------------

    def evaluate_sample(self, question: dict) -> dict:
        """Run generation for one AMBER sample.

        Args:
            question: dict with keys "id" (or "question_id"), "text" (or "query"),
                      "image" (filename).

        Returns:
            dict with "id" and "response".
        """
        from llava.constants import (
            IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
        )
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

        qid = question.get("question_id", question.get("id"))
        qs = question.get("text", question.get("query"))
        image_file = question["image"]

        # Build prompt
        if self._model.config.mm_use_im_start_end:
            qs_model = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + \
                       DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs_model = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs_model)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).cuda()

        # Load image
        image = Image.open(os.path.join(self.image_folder, image_file))
        image_tensor = self._image_processor.preprocess(
            image, return_tensors='pt'
        )['pixel_values'][0]

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self._tokenizer, input_ids
        )

        # Reset hooks for this sample
        if self._hook_loggers:
            for logger in self._hook_loggers.values():
                logger.reinit()

        # Generate
        self._model.eval()
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        input_len = input_ids.shape[1]
        output = self._tokenizer.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )[0].strip()
        if output.endswith(stop_str):
            output = output[:-len(stop_str)].strip()

        return {"id": qid, "response": output}

    # ------------------------------------------------------------------
    # Full evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        image_folder: str,
        question_file: str,
        answers_file: str,
    ):
        """Run full AMBER evaluation over all samples.

        Args:
            image_folder: path to AMBER images.
            question_file: path to AMBER questions JSON.
            answers_file: path to save model responses (JSONL).
        """
        self.image_folder = image_folder

        # Load model + hooks
        self.load_model()
        self.setup_hooks()

        # Load questions
        with open(question_file, "r") as f:
            questions = json.load(f)
        print(f"[ADAPT] Loaded {len(questions)} questions from {question_file}")

        # Prepare output
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        out = open(answers_file, "w")

        for item in tqdm(questions, desc="AMBER eval"):
            try:
                result = self.evaluate_sample(item)
                out.write(json.dumps(result) + "\n")
                out.flush()
            except Exception as e:
                print(f"[ADAPT] Error on sample {item.get('id', '?')}: {e}")
                out.write(json.dumps({
                    "id": item.get("id", item.get("question_id", "unknown")),
                    "response": "",
                    "error": str(e),
                }) + "\n")
                out.flush()

        out.close()
        print(f"[ADAPT] Results saved to {answers_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_amber_eval():
    parser = argparse.ArgumentParser(
        description="AMBER benchmark evaluation with ADAPT"
    )
    # Model
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    # Data
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, default="./amber_results.jsonl")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    # ADAPT
    parser.add_argument("--use-adapt", action="store_true", default=True)
    parser.add_argument("--no-adapt", dest="use_adapt", action="store_false")
    parser.add_argument("--modify-attention", action="store_true", default=True)
    parser.add_argument("--no-modify", dest="modify_attention", action="store_false")
    parser.add_argument("--initial-phase-calls", type=int, default=5)
    parser.add_argument("--mode", type=str, default="quality",
                        choices=["hhi", "quality"])
    parser.add_argument("--adaptive-k1", type=float, default=0.4)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices, e.g. '0,10,20,31'")

    args = parser.parse_args()

    set_seed(args.seed)

    layer_indices = None
    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(",")]

    evaluator = AMBEREvaluator(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        use_adapt=args.use_adapt,
        modify_attention=args.modify_attention,
        mode=args.mode,
        initial_phase_calls=args.initial_phase_calls,
        adaptive_k1=args.adaptive_k1,
        layer_indices=layer_indices,
    )

    evaluator.evaluate(
        image_folder=args.image_folder,
        question_file=args.question_file,
        answers_file=args.answers_file,
    )


if __name__ == "__main__":
    run_amber_eval()
