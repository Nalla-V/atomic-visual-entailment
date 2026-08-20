"""Meta-feature extraction for AVE-LS."""

import json
import math
import os
from collections import Counter

import jsonlines
import numpy as np

from src import config

LABELS = config.FINAL_LABELS
LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}
ID_TO_LABEL = {i: lab for lab, i in LABEL_TO_ID.items()}

FILE_CANDIDATES = {
    "qwen_baseline_simple": ["baseline_simple_train.jsonl"],
    "qwen_baseline_structured": ["baseline_structured_train.jsonl"],
    "qwen_joint_simple": ["atomic_joint_simple_train.jsonl"],
    "qwen_joint_structured": ["atomic_joint_structured_train.jsonl"],
    "qwen_self_decompose_simple": ["self_decompose_simple_train.jsonl"],
    "qwen_self_decompose_structured": ["self_decompose_structured_train.jsonl"],
    "internvl_baseline_simple": ["baseline_simple_train.jsonl"],
    "internvl_baseline_structured": ["baseline_structured_train.jsonl"],
    "internvl_joint_simple": ["atomic_joint_simple_train.jsonl"],
    "internvl_joint_structured": ["atomic_joint_structured_train.jsonl"],
    "internvl_self_decompose_simple": ["self_decompose_simple_train.jsonl"],
    "internvl_self_decompose_structured": ["self_decompose_structured_train.jsonl"],
}

FEATURE_VARIANTS = {
    # Full-hypothesis prediction only: 2 VLMs x 2 prompt styles = K=4.
    "baseline_only": [
        "qwen_baseline_simple", "qwen_baseline_structured",
        "internvl_baseline_simple", "internvl_baseline_structured",
    ],
    # Atomic-only simple-prompt pool: 2 VLMs x 2 atomic methods = K=4.
    "simple_atomic": [
        "qwen_joint_simple", "qwen_self_decompose_simple",
        "internvl_joint_simple", "internvl_self_decompose_simple",
    ],
    # Atomic-only structured-prompt pool: 2 VLMs x 2 atomic methods = K=4.
    "structured_atomic": [
        "qwen_joint_structured", "qwen_self_decompose_structured",
        "internvl_joint_structured", "internvl_self_decompose_structured",
    ],
    # Full prediction pool: 2 VLMs x 2 prompt styles x 3 methods = K=12.
    "full_12_methods": [
        "qwen_baseline_simple", "qwen_baseline_structured",
        "qwen_joint_simple", "qwen_joint_structured",
        "qwen_self_decompose_simple", "qwen_self_decompose_structured",
        "internvl_baseline_simple", "internvl_baseline_structured",
        "internvl_joint_simple", "internvl_joint_structured",
        "internvl_self_decompose_simple", "internvl_self_decompose_structured",
    ],
}

ALL_METHOD_KEYS = sorted(set(m for ms in FEATURE_VARIANTS.values() for m in ms))

VARIANT_DISPLAY_ORDER = ["baseline_only", "simple_atomic",
                         "structured_atomic", "full_12_methods"]

VARIANT_DISPLAY_NAMES = {
    "baseline_only": "Full hypothesis",
    "simple_atomic": "Simple atomic",
    "structured_atomic": "Structured atomic",
    "full_12_methods": "Full prediction pool",
}


def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(v).strip() for v in x if str(v).strip())
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()


def normalize_label(x):
    text = safe_text(x).lower().strip()
    if text in LABEL_TO_ID:
        return text
    if "entailment" in text or "entailed" in text or "support" in text or "supported" in text:
        return "entailment"
    if "contradiction" in text or "contradict" in text or "conflict" in text or "conflicting" in text:
        return "contradiction"
    if "neutral" in text or "uncertain" in text or "unsupported" in text or "insufficient" in text:
        return "neutral"
    for lab in LABELS:
        if lab in text:
            return lab
    return "neutral"


def make_key_from_values(img_id, hypo, gold):
    return (safe_text(img_id), safe_text(hypo), normalize_label(gold))


def make_key(row):
    return make_key_from_values(
        row.get("Flickr30K_ID", ""),
        row.get("hypothesis", row.get("sentence2", "")),
        row.get("annotator_label", row.get("gold", "")),
    )


def read_jsonl(path):
    with jsonlines.open(path, "r") as reader:
        return [row for row in reader]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with jsonlines.open(path, "w") as writer:
        for row in rows:
            writer.write(row)


def resolve_file(folder, candidates):
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find any of {candidates} in {folder}")


def ensure_atom_texts(atoms):
    if not isinstance(atoms, list):
        return []
    out = []
    for atom in atoms:
        if isinstance(atom, str):
            text = safe_text(atom)
        elif isinstance(atom, dict):
            text = safe_text(atom.get("atom_text", atom.get("atom",
                             atom.get("claim", atom.get("fact", "")))))
        else:
            text = ""
        if text:
            out.append(text)
    return out


def word_count(text):
    return len(safe_text(text).split())


def atom_bucket(n):
    if n <= 1:
        return "1 atom"
    if n == 2:
        return "2 atoms"
    if n == 3:
        return "3 atoms"
    return "4+ atoms"


def scores_to_probs(scores):
    if not isinstance(scores, dict):
        return {lab: 1.0 / len(LABELS) for lab in LABELS}

    vals = []
    for lab in LABELS:
        try:
            vals.append(float(scores.get(lab, 0.0)))
        except Exception:
            vals.append(0.0)

    arr = np.array(vals, dtype=float)
    if np.any(np.isnan(arr)) or np.any(arr < 0) or arr.sum() <= 0:
        return {lab: 1.0 / len(LABELS) for lab in LABELS}

    arr = arr / arr.sum()
    return {LABELS[i]: float(arr[i]) for i in range(len(LABELS))}


def argmax_label(scores):
    return max(scores, key=scores.get)


def confidence_diag(scores):
    probs = scores_to_probs(scores)
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_prob = ordered[0]
    second_label, second_prob = ordered[1]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs.values())
    return {
        "confidence_score": float(top_prob),
        "margin": float(top_prob - second_prob),
        "entropy": float(entropy),
        "normalized_entropy": entropy / math.log(len(LABELS)),
        "top_label": top_label,
        "second_label": second_label,
    }


def label_one_hot(label, prefix):
    lab = normalize_label(label)
    return {f"{prefix}_is_{x}": 1 if lab == x else 0 for x in LABELS}


def first_existing_dict(row, candidate_keys):
    for key in candidate_keys:
        value = row.get(key, {})
        if isinstance(value, dict) and value:
            return value
    return {}


def find_result_dict_recursively(obj):
    if not isinstance(obj, dict):
        return {}
    has_label = any(k in obj for k in ["prediction", "label", "final_label", "answer"])
    has_score = any(k in obj for k in ["scores", "probabilities", "class_scores"])
    if has_label or has_score:
        return obj
    for value in obj.values():
        if isinstance(value, dict):
            found = find_result_dict_recursively(value)
            if found:
                return found
    return {}


def get_scores_from_result(res):
    scores = (res.get("scores") or res.get("probabilities")
              or res.get("class_scores") or res.get("score_dict") or {})
    return scores_to_probs(scores)


SELF_DECOMPOSE_KEYS = [
    "self_decompose_results", "self_decomposition_results",
    "self_decompose_result", "self_decomposition_result",
    "self_decompose_final_results", "self_decomposition_final_results",
    "atomic_self_decompose_results", "self_decomposed_results",
]


def _extract_common(res):
    scores = get_scores_from_result(res)
    pred = normalize_label(res.get("prediction", res.get("label",
                           res.get("final_label", ""))))
    score_pred = normalize_label(res.get("score_prediction",
                                 res.get("score_label", argmax_label(scores))))
    diag = confidence_diag(scores)
    return {
        "prediction": pred,
        "score_prediction": score_pred,
        "scores": scores,
        "parse_ok": bool(res.get("parse_ok", True)),
        "reason": safe_text(res.get("reason", "")),
        "raw_result": res,
        **diag,
    }


def extract_baseline(row):
    res = row.get("full_hypothesis_results", {}) or {}
    return _extract_common(res if isinstance(res, dict) else {})


def extract_joint(row):
    res = row.get("joint_atom_results", {}) or {}
    return _extract_common(res if isinstance(res, dict) else {})


def extract_self_decompose(row):
    res = first_existing_dict(row, SELF_DECOMPOSE_KEYS)
    if not res:
        res = find_result_dict_recursively(row)
    if not res:
        res = row
    if not isinstance(res, dict):
        res = {}

    scores = get_scores_from_result(res)
    pred = normalize_label(res.get("prediction", res.get("label",
                           res.get("final_label", res.get("answer",
                           res.get("model_prediction", ""))))))
    score_pred = normalize_label(res.get("score_prediction",
                                 res.get("score_label", res.get("score_pred",
                                 argmax_label(scores)))))
    diag = confidence_diag(scores)
    return {
        "prediction": pred,
        "score_prediction": score_pred,
        "scores": scores,
        "parse_ok": bool(res.get("parse_ok", True)),
        "reason": safe_text(res.get("reason", "")),
        "raw_result": res,
        **diag,
    }


def method_kind(method_key):
    if "baseline" in method_key:
        return "baseline"
    if "joint" in method_key:
        return "joint"
    if "self_decompose" in method_key:
        return "self_decompose"
    raise ValueError(f"Cannot infer method kind from {method_key}")


EXTRACTORS = {"baseline": extract_baseline, "joint": extract_joint,
              "self_decompose": extract_self_decompose}


def load_dataset_metadata(path):
    rows = read_jsonl(path)
    out = {}
    for i, row in enumerate(rows):
        img_id = safe_text(row.get("Flickr30K_ID", ""))
        hypo = safe_text(row.get("hypothesis", row.get("sentence2", "")))
        gold = normalize_label(row.get("annotator_label", row.get("gold", "")))
        atoms = ensure_atom_texts(row.get("atomic_facts", row.get("raw_atoms", [])))
        if not atoms:
            atoms = [hypo]

        out[make_key_from_values(img_id, hypo, gold)] = {
            "order": i,
            "Flickr30K_ID": img_id,
            "hypothesis": hypo,
            "gold": gold,
            "num_atoms": len(atoms),
            "atom_bucket": atom_bucket(len(atoms)),
            "hypothesis_word_count": word_count(hypo),
            "total_atom_word_count": sum(word_count(a) for a in atoms),
        }
    return out


def load_prediction_map(path, kind):
    rows = read_jsonl(path)
    extractor = EXTRACTORS[kind]
    return {make_key(row): extractor(row) for row in rows}


def vote_summary(labels):
    """Majority indicators mean 'has the highest vote count', so a tie sets
    both label indicators and the tie indicator."""
    labels = [normalize_label(x) for x in labels]
    counts = Counter(labels)
    if not counts:
        return {"unique_count": 0, "largest_vote_count": 0, "top_labels": [],
                "has_strict_majority": 0, "has_top_vote_tie": 0}

    largest = max(counts.values())
    top_labels = [lab for lab in LABELS if counts.get(lab, 0) == largest]
    return {
        "unique_count": len(set(labels)),
        "largest_vote_count": int(largest),
        "top_labels": top_labels,
        "has_strict_majority": int(largest > len(labels) / 2),
        "has_top_vote_tie": int(len(top_labels) > 1),
    }


def build_feature_row(metadata, method_outputs, method_keys):
    feats = {}

    num_atoms = int(metadata.get("num_atoms", 1))
    feats["num_atoms"] = num_atoms
    feats["hypothesis_word_count"] = int(metadata.get("hypothesis_word_count", 0))
    feats["total_atom_word_count"] = int(metadata.get("total_atom_word_count", 0))
    feats["atom_bucket_1"] = 1 if num_atoms <= 1 else 0
    feats["atom_bucket_2"] = 1 if num_atoms == 2 else 0
    feats["atom_bucket_3"] = 1 if num_atoms == 3 else 0
    feats["atom_bucket_4plus"] = 1 if num_atoms >= 4 else 0

    preds, score_preds = [], []

    for method_key in method_keys:
        out = method_outputs[method_key]
        pred = normalize_label(out.get("prediction", "neutral"))
        score_pred = normalize_label(out.get("score_prediction", pred))
        scores = scores_to_probs(out.get("scores", {}))

        preds.append(pred)
        score_preds.append(score_pred)
        prefix = method_key

        feats.update(label_one_hot(pred, f"{prefix}_pred"))
        feats.update(label_one_hot(score_pred, f"{prefix}_score_pred"))

        for lab in LABELS:
            feats[f"{prefix}_score_{lab}"] = float(scores.get(lab, 0.0))

        diag = confidence_diag(scores)
        feats[f"{prefix}_highest_score"] = float(out.get("confidence_score", diag["confidence_score"]))
        feats[f"{prefix}_score_margin"] = float(out.get("margin", diag["margin"]))
        feats[f"{prefix}_score_entropy"] = float(out.get("entropy", diag["entropy"]))
        feats[f"{prefix}_normalized_score_entropy"] = float(
            out.get("normalized_entropy", diag["normalized_entropy"]))

        top_label = normalize_label(out.get("top_label", diag["top_label"]))
        second_label = normalize_label(out.get("second_label", diag["second_label"]))
        feats.update(label_one_hot(top_label, f"{prefix}_top"))
        feats.update(label_one_hot(second_label, f"{prefix}_second"))

        # Parse flags are excluded, keeping the selector on prediction, score,
        # score-vector, vote, agreement and metadata features.
        feats[f"{prefix}_pred_equals_score_pred"] = 1 if pred == score_pred else 0

    counts = Counter(preds)
    score_counts = Counter(score_preds)

    for lab in LABELS:
        feats[f"num_votes_{lab}"] = counts.get(lab, 0)
        feats[f"frac_votes_{lab}"] = counts.get(lab, 0) / len(preds) if preds else 0.0
        feats[f"num_score_votes_{lab}"] = score_counts.get(lab, 0)
        feats[f"frac_score_votes_{lab}"] = score_counts.get(lab, 0) / len(score_preds) if score_preds else 0.0

    pred_vs = vote_summary(preds)
    score_vs = vote_summary(score_preds)

    feats["num_unique_pred_labels"] = pred_vs["unique_count"]
    feats["all_methods_agree"] = 1 if pred_vs["unique_count"] == 1 else 0
    feats["has_disagreement"] = 1 if pred_vs["unique_count"] > 1 else 0
    feats["majority_count"] = pred_vs["largest_vote_count"]

    feats["num_unique_score_pred_labels"] = score_vs["unique_count"]
    feats["all_score_preds_agree"] = 1 if score_vs["unique_count"] == 1 else 0
    feats["has_score_pred_disagreement"] = 1 if score_vs["unique_count"] > 1 else 0
    feats["score_majority_count"] = score_vs["largest_vote_count"]

    for lab in LABELS:
        feats[f"majority_label_is_{lab}"] = 1 if lab in pred_vs["top_labels"] else 0
    feats["majority_type_is_majority"] = pred_vs["has_strict_majority"]
    feats["majority_type_is_tie"] = pred_vs["has_top_vote_tie"]

    for lab in LABELS:
        feats[f"score_majority_label_is_{lab}"] = 1 if lab in score_vs["top_labels"] else 0
    feats["score_majority_type_is_majority"] = score_vs["has_strict_majority"]
    feats["score_majority_type_is_tie"] = score_vs["has_top_vote_tie"]

    for i in range(len(method_keys)):
        for j in range(i + 1, len(method_keys)):
            a, b = method_keys[i], method_keys[j]
            feats[f"agree_{a}__{b}"] = int(
                normalize_label(method_outputs[a]["prediction"])
                == normalize_label(method_outputs[b]["prediction"]))
            feats[f"score_agree_{a}__{b}"] = int(
                normalize_label(method_outputs[a]["score_prediction"])
                == normalize_label(method_outputs[b]["score_prediction"]))

    return feats


def align_candidate_to_label(learned_label, method_outputs, method_keys):
    """Map a predicted label back to the candidate that best supports it."""
    learned_label = normalize_label(learned_label)

    matching = [k for k in method_keys
                if normalize_label(method_outputs[k].get("prediction", "neutral")) == learned_label]

    def confidence_key(method_key):
        out = method_outputs[method_key]
        scores = scores_to_probs(out.get("scores", {}))
        diag = confidence_diag(scores)
        return (float(out.get("confidence_score", diag["confidence_score"])),
                float(out.get("margin", diag["margin"])),
                float(scores.get(learned_label, 0.0)),
                -method_keys.index(method_key))

    if matching:
        return max(matching, key=confidence_key), "matched_label_highest_confidence"

    def fallback_key(method_key):
        out = method_outputs[method_key]
        scores = scores_to_probs(out.get("scores", {}))
        diag = confidence_diag(scores)
        return (float(scores.get(learned_label, 0.0)),
                float(out.get("confidence_score", diag["confidence_score"])),
                float(out.get("margin", diag["margin"])),
                -method_keys.index(method_key))

    return max(method_keys, key=fallback_key), "no_label_match_highest_score_for_learned_label"