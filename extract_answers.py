#!/usr/bin/env python3
"""
extract_final_answer.py

Add an `"extracted_answer"` field to every *.json file under ROOT_DIR by
calling the OpenAI / Azure OpenAI ChatCompletion API with a few-shot prompt.

Example
-------
python extract_final_answer.py /home/jovyan/workspace/bagel/MathVista-1000/testmini \
    --response_field response \
    --model gpt-4o-mini \
    --overwrite

Environment variables
---------------------
OPENAI_API_KEY            (for OpenAI)
OPENAI_API_BASE           (optional; set to Azure endpoint for Azure OpenAI)
AZURE_OPENAI_DEPLOYMENT   (optional; model deployment name)

Requires: pip install openai tqdm rich
"""
import argparse
import json
import logging
import os
import re
from pathlib import Path

from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
from rich.logging import RichHandler
from tqdm import tqdm

# ---------- few-shot prompt ---------------------------------------------------
DEMO_PROMPT = r"""
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?
Choices:
(A) 3/11
(B) 8/11
(C) 6/11
(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
""".strip()
# -----------------------------------------------------------------------------


def valid(extraction: str) -> bool:
    """Return True if extraction looks non-empty."""
    return bool(extraction and extraction.strip())


def quick_extract(text: str) -> str | None:
    """
    Cheap rule: look for
        The answer is "XYZ".
    """
    m = re.search(r'The answer is "?(.+?)"?\.?$', text, flags=re.I | re.M)
    return m.group(1).strip() if m else None


def build_prompt(problem_question: str, model_response: str) -> str:
    """Concatenate few-shot examples, question and response."""
    return (
        f"{DEMO_PROMPT}\n\n"
        f"{problem_question}\n\n"
        f"Model response: {model_response}\n\n"
        "Extracted answer: "
    )


def chat_completion(prompt: str, model: str) -> str:
    """Call ChatCompletion and return the assistant message."""
    resp = client.chat.completions.create(model=model,
    messages=[
        {"role": "user", "content": prompt},
    ],
    temperature=0.0,
    max_tokens=16)
    return resp.choices[0].message.content.strip()


def process_file(
    json_path: Path,
    question_field: str,
    response_field: str,
    model: str,
    overwrite: bool,
) -> bool:
    """
    Add extracted_answer to one json file.
    Returns True if updated.
    """
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Skip if key already exists and not overwriting
    if not overwrite and "extracted_answer" in data and valid(data["extracted_answer"]):
        return False

    question = data.get(question_field, "") or data.get("question", "")
    response = data.get(response_field, "")
    if response is None:
        logging.warning(f"[skip] {json_path} missing '{response_field}'")
        return False

    # first try rule-based
    extraction = quick_extract(response)
    if not extraction:
        # fall back to GPT
        prompt = build_prompt(question, response)
        try:
            extraction = chat_completion(prompt, model)
        except Exception as e:
            logging.error(f"OpenAI error on {json_path}: {e}")
            return False

    if not valid(extraction):
        logging.warning(f"[invalid] Could not extract answer from {json_path}")
        return False

    data["extracted_answer"] = extraction
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", help="directory tree containing sub-folders with *.json files")
    parser.add_argument("--question_field", default="query", help="json key holding the original question")
    parser.add_argument("--response_field", default="final_answer", help="json key holding the model response")
    parser.add_argument("--model", default="gpt-4o-mini", help="model (or Azure deployment name)")
    parser.add_argument("--overwrite", action="store_true", help="regenerate even if extracted_answer exists")
    args = parser.parse_args()

    # ---------- OpenAI credentials -------------------------------------------
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Set OPENAI_API_KEY (or configure Azure credentials) first.")
    # For Azure, you may need:
    # openai.api_type = "azure"
    # openai.api_base = os.environ["OPENAI_API_BASE"]
    # openai.api_version = "2023-05-15"  # example version
    # -------------------------------------------------------------------------

    files = list(Path(args.root_dir).rglob("*.json"))
    logging.info(f"{len(files)} JSON files found under {args.root_dir}")

    updated = 0
    for fp in tqdm(files, desc="extracting"):
        if process_file(
            fp,
            question_field=args.question_field,
            response_field=args.response_field,
            model=args.model,
            overwrite=args.overwrite,
        ):
            updated += 1

    logging.info(f"Finished. Updated {updated}/{len(files)} files.")


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    main()
