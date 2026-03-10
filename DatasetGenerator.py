"""
Build unified prompt dataset for energy experiments.

Sources:
- LLM-generated prompts from a local CSV
- HuggingFace datasets:
    - HuggingFaceH4/MATH-500
    - cais/mmlu
    - openai/openai_humaneval
    - crownelius/Opus-4.6-Reasoning-3300x
    - google/simpleqa-verified

Output columns:
- prompt              : str
- prompt_length       : int  (word count)
- length_bucket       : int  in {0,1,2,3} (quartiles by prompt_length)
- task_type           : int  in {1..6}
- complexity          : int  in {0,1,2}  (0=low, 1=medium, 2=high)
- output_length       : nullable (to be filled later after experiments)
- origin              : str  (e.g., "ChatGPT", "code_contests", ...)
"""

from typing import List, Optional
import pandas as pd
from datasets import load_dataset
import re

# =========================
# CONFIG
# =========================

# Path to generated prompts csv file.
# Set to None if you want to skip this for now.
LLM_PROMPTS_CSV: Optional[str] = "Data/AI_generated_prompts.csv"

# Where to save the final combined dataset
OUTPUT_CSV: str = "Data/dataset.csv"

# Random seed for reproducibility
RANDOM_SEED: int = 42
SAMPLES_PER_DATASET: int = 100

# The maximum length a prompt can be in the final dataset
# (longer candidate prompts are skipped)
MAX_PROMPT_WORDS = 250

# Task type legend:
# 1 = Casual / small talk
# 2 = Explanation / learning
# 3 = Writing / rewriting
# 4 = Coding / technical help
# 5 = Productivity / planning / structured
# 6 = Creative / storytelling

# HuggingFace dataset configs:
HF_DATASETS = [
    {
        "name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "subset": None,
        "prompt_fields": ["problem"],
        "task_type": 4,          # Coding / technical help (math)
        "complexity": 2,         # High complexity
        "origin": "MATH-500",
    },
    {
        "name": "cais/mmlu",
        "split": "test",
        "subset": "all",
        "prompt_fields": ["question"],
        "task_type": 2,          # Explanation / learning
        "complexity": 1,         # High complexity
        "origin": "MMLU",
    },
    {
        "name": "openai/openai_humaneval",
        "split": "test",
        "subset": None,
        "prompt_fields": ["prompt"],
        "task_type": 4,          # Coding / technical help
        "complexity": 2,         # High complexity
        "origin": "HumanEval",
    },
    {
        "name": "crownelius/Opus-4.6-Reasoning-3300x",
        "split": "train",
        "subset": None,
        "prompt_fields": ["problem"],
        "task_type": 2,          # Explanation / learning / reasoning
        "complexity": 2,         # High complexity
        "origin": "OpusReasoning",
    },
    {
        "name": "google/simpleqa-verified",
        "split": "eval",
        "subset": None,
        "prompt_fields": ["problem"],
        "task_type": 1,          # Casual / small talk-like Q&A
        "complexity": 0,
        "origin": "SimpleQA",
    },
]


# =========================
# HELPER METHODS
# =========================

def clean_prompt_text(text: str) -> str:
    """
    Normalize prompt text by:
    - Replacing all non-space whitespace with a space
    - Removing special characters beyond standard punctuation
    - Collapsing multiple spaces into one
    - Stripping leading/trailing spaces
    """

    if not isinstance(text, str):
        return ""

    # 1. Replace all whitespace except space with space
    # \s includes space, so we handle explicitly:
    text = re.sub(r"[^\S ]+", " ", text)

    # 2. Remove characters that are NOT:
    #    letters, numbers, space, or standard punctuation
    text = re.sub(r"[^a-zA-Z0-9 .,!?;:'\"()\-\[\]{}<>/\\@#$%^&*_+=|`~]", "", text)

    # 3. Collapse multiple spaces
    text = re.sub(r" +", " ", text)

    return text.strip()


def compute_word_count(text: str) -> int:
    """Token/word count based on whitespace splitting."""
    if not isinstance(text, str):
        return 0
    return len(text.strip().split())


def assign_length_buckets(lengths: pd.Series) -> pd.Series:
    """
    Assign each length to a bucket 0-3 so that approximately 25%
    of the dataset falls into each bucket.
    """
    q1, q2, q3 = lengths.quantile([0.25, 0.5, 0.75])

    def bucket(l: float) -> int:
        if l <= q1:
            return 0
        elif l <= q2:
            return 1
        elif l <= q3:
            return 2
        else:
            return 3

    return lengths.apply(bucket)


def extract_prompt_from_example(example: dict, candidates: List[str]) -> Optional[str]:
    """
    Try a list of candidate field names to get a prompt string from a HuggingFace example.
    Returns None if nothing works.
    """
    for field in candidates:
        if field in example and example[field] is not None:
            value = example[field]
            if isinstance(value, str):
                return value
    print(f"No prompt found for example {format(example)}")
    return None


def load_hf_prompts() -> pd.DataFrame:
    """
    Load exactly SAMPLES_PER_DATASET random prompts
    from each configured HuggingFace dataset.
    Deterministic via RANDOM_SEED.
    """
    records = []

    for cfg in HF_DATASETS:
        print(f"\nLoading HF dataset: {cfg['name']}")

        try:
            if cfg["subset"]:
                ds = load_dataset(
                    cfg["name"],
                    cfg["subset"],
                    split=cfg["split"],
                )
            else:
                ds = load_dataset(
                    cfg["name"],
                    split=cfg["split"],
                )
        except Exception as e:
            print(f"Failed to load {cfg['name']}: {e}")
            continue

        # Deterministic shuffle
        ds = ds.shuffle(seed=RANDOM_SEED)

        # Determine sample size safely
        sample_size = min(SAMPLES_PER_DATASET, len(ds))

        if sample_size < SAMPLES_PER_DATASET:
            print(
                f"  Warning: {cfg['name']} has only {len(ds)} rows. "
                f"Using {sample_size} instead of {SAMPLES_PER_DATASET}."
            )

        sampled_ds = ds.select(range(sample_size))

        for idx, ex in enumerate(sampled_ds):
            prompt_text = extract_prompt_from_example(
                ex, cfg["prompt_fields"]
            )
            # Check that some text was extracted; if none was, skip
            if not prompt_text:
                continue
            word_count = compute_word_count(prompt_text)
            # Skip prompts that exceed max length
            if word_count > MAX_PROMPT_WORDS:
                continue

            records.append(
                {
                    "prompt": str(prompt_text).strip(),
                    "prompt_length": word_count,
                    "task_type": cfg["task_type"],
                    "complexity": cfg["complexity"],
                    "output_length": None,
                    "origin": cfg["origin"]
                }
            )

        print(f"  -> Collected {sample_size} samples.")

    return pd.DataFrame.from_records(records)


def load_llm_prompts(csv_path: str) -> pd.DataFrame:
    """
    Load your manually annotated ChatGPT prompts from CSV.

    Expected columns:
      - prompt (str)
      - task_type (int in {1..6})
      - complexity (int in {0,1,2})
      - output_length (empty for now)
    """
    print(f"\nLoading ChatGPT prompts from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Ensure 'prompt' exists
    if "prompt" not in df.columns:
        raise ValueError("LLM prompts CSV must contain a 'prompt' column.")

    # Standardize/ensure columns
    if "prompt_length" not in df.columns:
        df["prompt_length"] = pd.NA
    if "task_type" not in df.columns:
        df["task_type"] = pd.NA
    if "complexity" not in df.columns:
        df["complexity"] = pd.NA
    if "output_length" not in df.columns:
        df["output_length"] = pd.NA

    df["origin"] = "ChatGPT"

    # Keep only the columns we care about at this stage
    df = df[["prompt", "prompt_length", "task_type", "complexity", "output_length", "origin"]]
    # Compute prompt length [saves us from having to recompute prompt length later]
    df["prompt_length"] = df["prompt"].apply(compute_word_count)

    print(f"  -> loaded {len(df)} ChatGPT prompts.")
    return df


# =========================
# MAIN PIPELINE
# =========================

def main():
    # 1. Load HF dataset prompts
    hf_df = load_hf_prompts()

    # 2. Load ChatGPT prompts
    if LLM_PROMPTS_CSV is not None:
        llm_df = load_llm_prompts(LLM_PROMPTS_CSV)
        combined = pd.concat([hf_df, llm_df], ignore_index=True)
    else:
        combined = None
        print("Could not find AI-generated prompts csv file.")

    # 3. Clean all prompts
    combined["prompt"] = combined["prompt"].apply(clean_prompt_text)

    # 4. Compute prompt_length (word count)


    # 5. Compute length_bucket so 25% of samples fall into each bucket
    combined["length_bucket"] = assign_length_buckets(combined["prompt_length"])

    # 6. Confirm correct ordering of columns for consistency
    combined = combined[
        [
            "prompt",
            "prompt_length",
            "length_bucket",
            "task_type",
            "complexity",
            "output_length",
            "origin"
        ]
    ]

    # 7. Save to CSV
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved combined dataset with {len(combined)} rows to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
