"""Paths, model names and shared settings. No logic here."""

import os

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.abspath("."))
INPUT_DIR = os.path.join(DATA_ROOT, "Input")
OUTPUT_DIR = os.path.join(DATA_ROOT, "Output")
DEMONS_JSON = os.path.join(INPUT_DIR, "demons.json")

HF_TOKEN = os.environ.get("HF_TOKEN") or None
HF_CACHE_DIR = os.environ.get("HF_HOME") or None

SPLITS = ("train", "dev", "test")

MAX_INPUT_TOKENS = 3200
MAX_NEW_TOKENS = 256
TOP_K_EXAMPLES = 4
PROGRESS_EVERY = {"train": 5000, "dev": 100, "test": 100}


def input_file(split):
    return os.path.join(INPUT_DIR, f"snli_ve_{split}.jsonl")


def output_file(stem, model_key, split, debug=False):
    suffix = "_debug" if debug else ""
    return os.path.join(OUTPUT_DIR, f"{stem}_{model_key}_{split}{suffix}.jsonl")


# role: "main" is the proposed pipeline, "ablation" is section 5.1 only.
DECOMPOSER_MODELS = {
    "qwen32": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "dtype": "bfloat16",
        "gated": False,
        "env": "ENV_NLI",
        "role": "main",
    },
    "qwen3": {
        "hf_id": "Qwen/Qwen3-8B",
        "dtype": "bfloat16",
        "gated": False,
        "disable_thinking": True,
        "env": "ENV_QWEN",
        "role": "ablation",
    },
    "llama": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "dtype": "float16",
        "gated": True,
        "env": "ENV_NLI",
        "role": "ablation",
    },
}


def models_with_role(role):
    return [k for k, v in DECOMPOSER_MODELS.items() if v.get("role") == role]