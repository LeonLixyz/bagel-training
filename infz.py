#!/usr/bin/env python3
"""
evaluate_interleaved_json.py

Batch-style evaluator for interleaved text-image reasoning models.
Modified for distributed evaluation across multiple GPUs.

Folder structure expected:
root_dir/
    00001/
        problem.json          # {"question": "...", "image": "problem_image_1.jpg"}
        problem_image_1.jpg
    00002/
        problem.json
        problem_image_1.png
    ...

The script walks every immediate sub-folder under root_dir, loads the JSON
metadata and image, runs the inference loop, and writes results back into the
same JSON file while saving generated images alongside it:

{
  "question": "...",
  "image": "problem_image_1.jpg",
  "reasoning": [
      {"text": "first thought", "image": "reasoning_image_1.png"},
      {"text": "second thought", "image": "reasoning_image_2.png"}
  ],
  "final_answer": "C"
}

Images are saved as reasoning_image_<n>.png in the same folder.
The JSON file is updated incrementally so you can monitor progress.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
import torch
from tqdm import tqdm

# ------- YOUR MODEL IMPORTS GO HERE ---------------------------------------- #
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens, pil_img2rgb
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from inferencer import InterleaveInferencer

import random
import numpy as np

# Set seed for reproducibility - same across all devices for evaluation
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------------------

def build_models(base_model_path: str, checkpoint_path: str, max_mem_per_gpu: str = "80GiB", gpu_id: Optional[int] = None):
    """Build and dispatch the Bagel model exactly as in the original script."""
    # --- LLM config ---
    llm_config = Qwen2Config.from_json_file(os.path.join(base_model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    # --- ViT config ---
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(base_model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    # --- VAE ---
    vae_model, vae_config = load_ae(local_path=os.path.join(base_model_path, "ae.safetensors"))

    # --- Bagel config ---
    bagel_cfg = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    # --- Build empty model skeleton ---
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_cfg)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    # --- Tokeniser & transforms ---
    tokenizer = Qwen2Tokenizer.from_pretrained(base_model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 512, 14)

    # --- Device mapping ---
    total_gpu = torch.cuda.device_count()
    if total_gpu == 0:
        raise RuntimeError("CUDA visible devices not found – run on a GPU machine.")

    # If a specific GPU is requested, constrain the device map to that GPU only.
    if gpu_id is not None:
        if gpu_id < 0 or gpu_id >= total_gpu:
            raise ValueError(f"Invalid gpu_id {gpu_id}; available GPUs are 0 to {total_gpu - 1}.")
        _max_memory = {gpu_id: max_mem_per_gpu}
    else:
        _max_memory = {i: max_mem_per_gpu for i in range(total_gpu)}

    device_map = infer_auto_device_map(
        model,
        max_memory=_max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        dtype=torch.bfloat16,
    )

    # Ensure certain small modules stay on the same (first) device for efficiency/stability.
    first_device_idx = gpu_id if gpu_id is not None else 0
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    first_device = device_map.get(same_device_modules[0], f"cuda:{first_device_idx}")
    for m in same_device_modules:
        device_map[m] = first_device

    # --- Load checkpoint ---
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,
        offload_buffers=False,
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()

    return model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids


def split_data_for_rank(problems: List[Path], rank: int, world_size: int) -> List[Path]:
    """Split the problems list evenly across world_size processes.
    
    Args:
        problems: List of problem paths
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes
    
    Returns:
        List of problems assigned to this rank
    """
    n_problems = len(problems)
    problems_per_rank = n_problems // world_size
    remainder = n_problems % world_size
    
    # Distribute remainder evenly among first 'remainder' ranks
    if rank < remainder:
        start_idx = rank * (problems_per_rank + 1)
        end_idx = start_idx + problems_per_rank + 1
    else:
        start_idx = rank * problems_per_rank + remainder
        end_idx = start_idx + problems_per_rank
    
    return problems[start_idx:end_idx]


# --------------------------------------------------------------------------------------
# Main evaluation logic per-problem
# --------------------------------------------------------------------------------------

def run_single_problem(
    problem_dir: Path,
    inferencer: InterleaveInferencer,
    inference_hyper: Dict[str, Any],
    system_prompt: Optional[str] = None,
    think: bool = False,
):
    """Execute reasoning for a single problem folder, updating its JSON file."""
    meta_path = problem_dir / "problem.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    prompt = meta["query"]
    # prompt = "Question: " + prompt
    img_name = meta.get("image", "problem_image_1.jpg")
    img_path = problem_dir / img_name
    image = Image.open(img_path).convert("RGB")

    current_input: List[Any] = [prompt, image]
    reasoning: List[Dict[str, Any]] = []
    iteration = 0
    max_iteration = 20

    while iteration < max_iteration:
        output = inferencer.interleave_inference(
            current_input,
            understanding_output=True,
            system_prompt=system_prompt,
            think=think,
            **inference_hyper,
        )

        raw_text: str = output[0]
        stop = ("<|vision_start|>" not in raw_text) or ("Final Answer" in raw_text)
        extracted_text = raw_text.split("<|im_end|>")[0].split("<|im_start|>")[1]
        # print(f"{problem_dir.name} | step {iteration} | {extracted_text[:80]}...")

        if stop:
            # Store final reasoning text without image
            reasoning.append({"text": extracted_text})
            meta["reasoning"] = reasoning
            # Extract answer token – naive split at "Final Answer:" if present
            if "Final Answer" in extracted_text:
                meta["final_answer"] = extracted_text.split("Final Answer")[-1].strip(" :\n")
            else:
                meta["final_answer"] = ""
            break

        # Generation continues – produce an image for current reasoning
        current_input_with_reasoning = current_input + [extracted_text]
        img_out = inferencer.interleave_inference(
            current_input_with_reasoning,
            system_prompt=system_prompt,
            think=think,
            **inference_hyper,
        )[1 if think else 0]

        img_filename = f"reasoning_image_{iteration+1}.png"
        img_full_path = problem_dir / img_filename
        img_out.save(img_full_path)

        reasoning.append({"text": extracted_text, "image": img_filename})
        meta["reasoning"] = reasoning  # update incrementally

        # Persist after each step so progress is visible on disk
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        current_input = current_input_with_reasoning + [img_out]
        iteration += 1

    if meta.get("final_answer", "") == "":
        # If no final answer was extracted, append a prompt to force the model to output one
        final_answer_prompt = "Final Answer: "
        output = inferencer.interleave_inference(
            current_input + [extracted_text, final_answer_prompt],
            understanding_output=True,
            system_prompt=system_prompt,
            think=think,
            **inference_hyper,
        )
        final_text = output[0]
        meta["final_answer"] = final_text

    # Final write
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta.get("final_answer", "")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch evaluator for interleaved CoT problems")
    parser.add_argument("--root_dir", help="Folder containing problem sub-directories")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory path")
    parser.add_argument("--checkpoint_file", default="model_bf16.safetensors")
    parser.add_argument("--base_model_path", default="/dev/shm/models/BAGEL-7B-MoT", help="Base model directory path")
    parser.add_argument("--device_mem", default="80GiB", help="Per-GPU memory constraint for device map")
    parser.add_argument("--gpu_id", type=int, default=None, help="Index of the GPU to use (default: auto-detect based on rank)")
    parser.add_argument("--rank", type=int, default=0, help="Process rank for distributed evaluation (0 to world_size-1)")
    parser.add_argument("--world_size", type=int, default=8, help="Total number of processes for distributed evaluation")
    args = parser.parse_args()

    # Auto-detect GPU ID based on rank if not specified
    if args.gpu_id is None:
        args.gpu_id = args.rank
        
    # Validate rank
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError(f"Rank must be between 0 and {args.world_size-1}, got {args.rank}")

    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)

    print(f"[Rank {args.rank}/{args.world_size}] Starting evaluation on GPU {args.gpu_id}")

    # Build models (this can take a few minutes)
    model, vae_model, tokenizer, vae_tf, vit_tf, new_token_ids = build_models(
        args.base_model_path, ckpt_path, args.device_mem, gpu_id=args.gpu_id
    )

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_tf,
        vit_transform=vit_tf,
        new_token_ids=new_token_ids,
    )

    inference_hyper = dict(
        do_sample=True,
        text_temperature=0.7,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval=[0.0, 1.0],
        timestep_shift=3.0,
        num_timesteps=50,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    ) 

    root = Path(args.root_dir)
    all_problems = sorted([p for p in root.iterdir() if p.is_dir()])
    
    # Split problems for this rank
    problems = split_data_for_rank(all_problems, args.rank, args.world_size)
    
    print(f"[Rank {args.rank}/{args.world_size}] Processing {len(problems)} out of {len(all_problems)} total problems")

    system_prompt = '''You are an AI reasoning assistant capable of step-by-step interleaved text and visual chain of thought. Think step by step and use visual aids to enhance your problem-solving. Provide your final conclusion clearly in the format of "Final Answer: <answer here>"'''

    start_time = time.time()
    answers = {}
    
    for p in tqdm(problems, desc=f"Rank {args.rank} processing", position=args.rank):
        try:
            ans = run_single_problem(p, inferencer, inference_hyper, system_prompt)
            answers[p.name] = ans
            # print(f"[Rank {args.rank}] [DONE] {p.name}: {ans}")
        except Exception as e:
            tqdm.write(f"[Rank {args.rank}] [ERROR] {p.name}: {e}")

    dur = time.time() - start_time
    print(f"[Rank {args.rank}] Processed {len(answers)} problems in {dur/60:.1f} min")

    # Write rank-specific summary file
    summary_path = root / f"results_summary_rank_{args.rank}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(answers, f, indent=2)
    print(f"[Rank {args.rank}] Summary written to {summary_path}")

    # If this is rank 0 and we're done, optionally merge all summaries
    if args.rank == 0:
        print(f"\n[Rank 0] To merge all results after all ranks complete, run:")
        print(f"python merge_results.py --root_dir {args.root_dir} --world_size {args.world_size}")


if __name__ == "__main__":
    main()