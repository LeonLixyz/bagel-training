#!/usr/bin/env python3
"""
parse_mmmu_fast.py

Faster version of MMMU converter with progress tracking and resume capability.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# All available MMMU subjects (configs)
ALL_MMMU_SUBJECTS = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 
    'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 
    'Clinical_Medicine', 'Computer_Science', 'Design', 
    'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 
    'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 
    'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology'
]


def save_images(example: Dict[str, Any], task_dir: Path) -> List[str]:
    """Save all images from the example and return their filenames."""
    saved_images = []
    
    # MMMU can have multiple images in different fields
    img_count = 0
    for i in range(1, 8):  # MMMU supports up to 7 images
        img_field = f"image_{i}"
        if img_field in example and example[img_field] is not None:
            img_count += 1
            img_filename = f"problem_image_{img_count}.png"
            img_path = task_dir / img_filename
            
            # Skip if already exists
            if img_path.exists():
                saved_images.append(img_filename)
                continue
            
            # Handle different image formats
            img_data = example[img_field]
            try:
                if isinstance(img_data, Image.Image):
                    img_data.save(img_path, optimize=True)
                elif isinstance(img_data, dict) and "bytes" in img_data:
                    Image.open(img_data["bytes"]).save(img_path, optimize=True)
                elif isinstance(img_data, str):
                    Image.open(img_data).save(img_path, optimize=True)
                else:
                    print(f"Warning: Unsupported image format for {img_field}")
                    continue
            except Exception as e:
                print(f"Error saving image {img_field}: {e}")
                continue
            
            saved_images.append(img_filename)
    
    return saved_images


def parse_options(options_str: str) -> List[str]:
    """Parse MMMU options string into a list."""
    if not options_str:
        return []
    
    if isinstance(options_str, list):
        return options_str
    
    # Extract options using regex
    pattern = r'\([A-Z]\)\s*([^(]*?)(?=\([A-Z]\)|$)'
    matches = re.findall(pattern, options_str)
    return [match.strip() for match in matches if match.strip()]


def build_question_text(example: Dict[str, Any]) -> str:
    """Build the complete question text including choices if present."""
    question = example.get("question", "").strip()
    
    # Get options/choices
    options = []
    if "options" in example and example["options"]:
        options = parse_options(example["options"])
    elif "choices" in example and example["choices"]:
        options = example["choices"]
    
    # Append choices to question if they exist
    if options:
        question += "\n\nChoices:"
        for i, opt in enumerate(options):
            question += f"\n({chr(65+i)}) {opt}"
    
    return question


def process_single_example(example: Dict[str, Any], subject: str, split_name: str, 
                          output_dir: Path, idx: int) -> bool:
    """Process a single example. Returns True if successful."""
    try:
        # Create unique ID with subject prefix
        pid = example.get("id") or example.get("question_id") or f"{subject}_{split_name}_{idx}"
        pid = f"{subject}_{pid}"
        
        task_dir = output_dir / pid
        
        # Skip if already processed
        if (task_dir / "problem.json").exists():
            return True
        
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all images
        saved_images = save_images(example, task_dir)
        
        # Build metadata
        meta = {
            "question": build_question_text(example),
            "answer": example.get("answer", ""),
            "metadata": {
                "subject": subject,
                "subfield": example.get("subfield", ""),
                "topic": example.get("topic", ""),
                "question_type": example.get("question_type", ""),
                "answer_type": example.get("answer_type", ""),
                "split": split_name
            }
        }
        
        # Add image reference(s)
        if len(saved_images) == 1:
            meta["image"] = saved_images[0]
        elif len(saved_images) > 1:
            meta["images"] = saved_images
            meta["image"] = saved_images[0]
        
        # Add choices if available
        if "options" in example and example["options"]:
            meta["choices"] = parse_options(example["options"])
        elif "choices" in example and example["choices"]:
            meta["choices"] = example["choices"]
        
        # Write problem.json
        with (task_dir / "problem.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"\nError processing example {idx}: {e}")
        return False


def process_subject_split(subject: str, split_name: str, out_root: Path, 
                         max_workers: int = 4, skip_existing: bool = True):
    """Process a specific subject and split combination with parallel processing."""
    try:
        # Load the dataset with subject as config
        print(f"Loading {subject} [{split_name}]...")
        ds = load_dataset("MMMU/MMMU", subject, split=split_name)
        print(f"Processing {subject} [{split_name}] – {len(ds)} examples")
    except Exception as e:
        print(f"Error loading {subject} [{split_name}]: {e}")
        return 0
    
    # Count existing if skip_existing is True
    existing_count = 0
    if skip_existing:
        for idx, ex in enumerate(ds):
            pid = ex.get("id") or ex.get("question_id") or f"{subject}_{split_name}_{idx}"
            pid = f"{subject}_{pid}"
            if (out_root / pid / "problem.json").exists():
                existing_count += 1
        
        if existing_count == len(ds):
            print(f"  ✓ All {len(ds)} examples already processed, skipping...")
            return len(ds)
        elif existing_count > 0:
            print(f"  Found {existing_count} existing examples, processing remaining...")
    
    # Process examples with progress bar
    successful = 0
    with tqdm(total=len(ds), desc=f"{subject}", unit="examples") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_single_example, ex, subject, split_name, out_root, idx): idx 
                for idx, ex in enumerate(ds)
            }
            
            # Process completed tasks
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                pbar.update(1)
    
    print(f"  ✓ Completed {subject}: {successful}/{len(ds)} examples processed successfully")
    return successful


def main():
    parser = argparse.ArgumentParser(description="Convert MMMU to folder-style problems (fast version)")
    parser.add_argument("--output_root", required=True,
                        help="Destination directory for converted data")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["test"],
        help="Dataset splits to convert (default: validation and test)",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help=f"Specific subjects to convert. Available: {', '.join(ALL_MMMU_SUBJECTS)}. If None, converts all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for processing (default: 4)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Reprocess existing examples instead of skipping them",
    )
    args = parser.parse_args()
    
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    
    # Determine which subjects to process
    subjects_to_process = args.subjects if args.subjects else ALL_MMMU_SUBJECTS
    
    # Validate subjects
    invalid_subjects = [s for s in subjects_to_process if s not in ALL_MMMU_SUBJECTS]
    if invalid_subjects:
        print(f"Warning: Invalid subjects will be skipped: {invalid_subjects}")
        subjects_to_process = [s for s in subjects_to_process if s in ALL_MMMU_SUBJECTS]
    
    if not subjects_to_process:
        print("No valid subjects to process!")
        return
    
    # Process each split and subject combination
    total_processed = 0
    start_time = time.time()
    
    for split in args.splits:
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing split: {split}")
        print("=" * 50)
        
        for subject in subjects_to_process:
            count = process_subject_split(
                subject, split, split_dir, 
                max_workers=args.workers,
                skip_existing=not args.no_skip_existing
            )
            total_processed += count
    
    elapsed = time.time() - start_time
    print(f"\n✓ MMMU conversion complete in {elapsed:.1f} seconds")
    print(f"Total problems processed: {total_processed}")
    
    # Print summary by split
    print("\nSummary by split:")
    for split in args.splits:
        split_dir = out_root / split
        if split_dir.exists():
            num_problems = len(list(split_dir.glob("*/problem.json")))
            print(f"  {split}: {num_problems} problems")


if __name__ == "__main__":
    main()