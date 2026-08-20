"""The K=12 candidate prediction pool: 2 VLMs x 3 methods x 2 prompt styles."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from src import config
from src.selection.common import (
    make_key,
    normalize_label,
    read_jsonl,
    score_argmax,
    score_margin,
    scores_to_probs,
)

VLMS = [("qwen", "Qwen3-VL-8B"), ("internvl", "InternVL3-8B")]

METHODS = [
    ("full_hypothesis", "Full hypothesis", "baseline_{prompt}_{split}.jsonl",
     "full_hypothesis_results", "fh"),
    ("atomic_prediction", "Atomic prediction", "atomic_joint_{prompt}_{split}.jsonl",
     "joint_atom_results", "atomic"),
    ("self_decomposed_atomic", "Self-decomposed atomic prediction",
     "self_decompose_{prompt}_{split}.jsonl", "self_decompose_results", "sd"),
]

PROMPTS = ["simple", "structured"]

# Folder suffix per split. Training used the raw prediction folders; dev and
# test used the cleaned ones.
FOLDER_SUFFIX = {"train": "_predictions_v2",
                 "dev": "_predictions_clean_v2",
                 "test": "_predictions_clean_v2"}

VLM_FOLDER = {"qwen": "qwen3", "internvl": "internvl"}


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    vlm: str
    vlm_display: str
    method_family: str
    method_display: str
    prompt: str
    file_name: str
    result_key: str


@dataclass
class SplitData:
    split: str
    specs: List[CandidateSpec]
    predictions: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]]
    keys: List[Tuple[str, str, str]]


def candidate_folder(split, vlm):
    return os.path.join(
        config.OUTPUT_DIR, f"{split}_dataset",
        VLM_FOLDER[vlm] + FOLDER_SUFFIX[split],
    )


def build_candidate_specs(split):
    specs = []
    for vlm, vlm_display in VLMS:
        for family, display, template, result_key, short in METHODS:
            for prompt in PROMPTS:
                specs.append(CandidateSpec(
                    candidate_id=f"{vlm}_{short}_{prompt}",
                    vlm=vlm,
                    vlm_display=vlm_display,
                    method_family=family,
                    method_display=display,
                    prompt=prompt,
                    file_name=template.format(prompt=prompt, split=split),
                    result_key=result_key,
                ))
    return specs


def get_result_dict(row, result_key):
    result = row.get(result_key, {})
    if isinstance(result, dict) and result:
        return result

    if result_key == "self_decompose_results":
        for key in ["self_decomposition_results", "self_decompose_result",
                    "self_decomposition_result", "self_decompose_final_results",
                    "self_decomposition_final_results",
                    "atomic_self_decompose_results", "self_decomposed_results"]:
            result = row.get(key, {})
            if isinstance(result, dict) and result:
                return result
    return {}


def extract_prediction(row, spec):
    key = make_key(row)
    result = get_result_dict(row, spec.result_key)
    scores = scores_to_probs(result.get("scores", {}))

    generated_label = normalize_label(
        result.get("prediction",
                   result.get("label",
                              result.get("final_label",
                                         result.get("model_prediction", ""))))
    )
    score_label = (normalize_label(result.get("score_prediction", ""))
                   if result.get("score_prediction", "") else score_argmax(scores))

    try:
        margin = float(result.get("margin", score_margin(scores)))
    except Exception:
        margin = score_margin(scores)

    return {
        "key": key,
        "Flickr30K_ID": key[0],
        "hypothesis": key[1],
        "gold": key[2],
        "candidate_id": spec.candidate_id,
        "vlm": spec.vlm,
        "vlm_display": spec.vlm_display,
        "method_family": spec.method_family,
        "method_display": spec.method_display,
        "prompt": spec.prompt,
        "generated_label": generated_label,
        "score_label": score_label,
        "margin": margin,
        "score_entailment": scores["entailment"],
        "score_neutral": scores["neutral"],
        "score_contradiction": scores["contradiction"],
    }


def check_required_files(splits):
    missing = []
    for split in splits:
        for spec in build_candidate_specs(split):
            path = os.path.join(candidate_folder(split, spec.vlm), spec.file_name)
            if not os.path.exists(path):
                missing.append(path)
    return missing


def load_split(split, verbose=True):
    specs = build_candidate_specs(split)
    predictions = {}
    key_order = {}

    if verbose:
        print(f"\nLoading {split} split", flush=True)

    for spec in specs:
        path = os.path.join(candidate_folder(split, spec.vlm), spec.file_name)
        rows = read_jsonl(path)
        candidate_map = {}

        for idx, row in enumerate(rows):
            record = extract_prediction(row, spec)
            candidate_map[record["key"]] = record
            if record["key"] not in key_order:
                key_order[record["key"]] = idx

        predictions[spec.candidate_id] = candidate_map
        if verbose:
            print(f"  {spec.vlm:7s} | {spec.method_family:24s} | "
                  f"{spec.prompt:10s} | rows={len(rows):6d}", flush=True)

    key_sets = [set(m.keys()) for m in predictions.values()]
    common_keys = set.intersection(*key_sets) if key_sets else set()
    all_keys = set.union(*key_sets) if key_sets else set()

    if verbose and len(common_keys) != len(all_keys):
        print(f"  WARNING: using common key intersection: "
              f"{len(common_keys)} / {len(all_keys)}", flush=True)

    keys = sorted(common_keys, key=lambda k: key_order.get(k, 10**12))
    if verbose:
        print(f"  Common rows used: {len(keys)}", flush=True)

    return SplitData(split=split, specs=specs, predictions=predictions, keys=keys)


def candidate_ids(specs, vlm=None, method_families=None, prompts=None):
    out = []
    for spec in specs:
        if vlm is not None and spec.vlm != vlm:
            continue
        if method_families is not None and spec.method_family not in method_families:
            continue
        if prompts is not None and spec.prompt not in prompts:
            continue
        out.append(spec.candidate_id)
    return out