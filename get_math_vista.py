#!/usr/bin/env python3
"""
parse_mathvista.py

Convert the MathVista dataset (https://huggingface.co/datasets/AI4Math/MathVista)
into the folder‑per‑problem layout expected by evaluate_interleaved_json.py.

Output structure:
root_dir/
    <id>/
        problem.json            # metadata & ground‑truth answer
        problem_image_1.<ext>

`problem.json` schema:
{
  "question": "...",                # question text (with choices appended if present)
  "image": "problem_image_1.png",  # relative filename of the main image
  "choices": ["A", "B", "C", ...],  # optional
  "answer": "C",                    # ground‑truth answer
  "question_type": "...",
  "answer_type": "...",
  "metadata": {...}
}

Usage:
    python parse_mathvista.py /path/to/output_root

The script downloads the dataset via the 🤗 Datasets library (cached locally on first run).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from PIL import Image

IMAGE_NAME = "problem_image_1.png"  # unify extension regardless of original


def save_image(example: Dict[str, Any], out_path: Path):
    """Save the decoded_image (or image) column as PNG."""
    if "decoded_image" in example and example["decoded_image"] is not None:
        pil_img = example["decoded_image"]
        if isinstance(pil_img, Image.Image):
            pil_img.save(out_path)
            return
    # fallback – datasets.Image() object or file path
    img_data = example["image"]
    if isinstance(img_data, dict) and "bytes" in img_data:
        # datasets.Image stored as dict { 'bytes': ..., 'path': None }
        Image.open(img_data["bytes"]).save(out_path)
    elif isinstance(img_data, str):
        Image.open(img_data).save(out_path)
    else:
        raise ValueError("Unsupported image field format – please inspect example structure.")


def build_question_text(example: Dict[str, Any]) -> str:
    q = example["question"].strip()
    choices = example.get("choices")
    if choices:
        # MathVista choices are typically strings like ["A", "B", ...] or dicts
        if isinstance(choices, list):
            choice_lines = [f"({chr(65+i)}) {c}" for i, c in enumerate(choices)]
        else:
            # already (A), (B) keys
            choice_lines = [f"{k}: {v}" for k, v in choices.items()]
        q += "\n\nChoices:\n" + "\n".join(choice_lines)
    return q


def process_split(split_name: str, out_root: Path):
    ds = load_dataset("AI4Math/MathVista", split=split_name)
    print(f"Processing {split_name} – {len(ds)} examples")

    for idx, ex in enumerate(ds):
        # --- robust problem-id lookup ---
        pid = ex.get("pid") or ex.get("id") or f"{split_name}_{idx}"
        task_dir = out_root / pid
        task_dir.mkdir(parents=True, exist_ok=True)

        # ---- save image ----
        img_path = task_dir / IMAGE_NAME
        save_image(ex, img_path)

        # ---- build problem.json ----
        meta = {
            "question": build_question_text(ex),
            "image": IMAGE_NAME,
            "choices": ex.get("choices", []),
            "answer": ex.get("answer", ""),
            "query": ex.get("query", ""),
        }
        with (task_dir / "problem.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert MathVista to folder-style problems")
    parser.add_argument("--output_root", required=True,
                        help="Destination directory for converted data")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["testmini", "test"],   # MathVista provides these two splits
        help="Dataset splits to convert (default: testmini and test)",
    )
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        process_split(split, out_root / split)

    print("✓ Conversion complete.")


if __name__ == "__main__":
    main()

# from datasets import load_dataset, DownloadConfig

# cfg = DownloadConfig(
#     cache_dir="/mnt/data/hf_cache",   # any fast disk
#     resume_download=True,
#     max_retries=20,
# )

# ds = load_dataset(
#     "AI4Math/MathVista",
#     split="testmini",               # or "train" / "validation" / "test"
#     download_config=cfg,
# )
