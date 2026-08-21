"""
result_candidate_prediction_analysis.py

Purpose
-------
Create the analysis outputs for Section 5.2 Candidate Prediction Results.

The script reads the dev/test prediction files for Qwen3-VL-8B and InternVL3-8B,
evaluates generated-label and score-based-label outputs, and writes figures/tables.

"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix


# ============================================================
# 1. CONSTANTS
# ============================================================

LABELS = ["entailment", "neutral", "contradiction"]
LABEL_TO_SHORT = {"entailment": "E", "neutral": "N", "contradiction": "C"}
BUCKETS = ["1 atom", "2 atoms", "3 atoms", "4+ atoms"]

SPLITS = ["dev", "test"]
PROMPTS = ["simple", "structured"]

VLM_CONFIGS = {
    "qwen": {
        "display": "Qwen3-VL-8B",
        "short": "Qwen",
        "folder": "qwen3_predictions_clean_v2",
    },
    "internvl": {
        "display": "InternVL3-8B",
        "short": "InternVL",
        "folder": "internvl_predictions_clean_v2",
    },
}

METHOD_CONFIGS = {
    "full_hypothesis": {
        "display": "Full-hypothesis",
        "short": "Full-hypothesis",
        "file_prefix": "baseline",
        "result_key": "full_hypothesis_results",
        "fusion_pool": "Yes",
        "marker": "o",
    },
    "atomic_prediction": {
        "display": "Atomic prediction",
        "short": "Atomic prediction",
        "file_prefix": "atomic_joint",
        "result_key": "joint_atom_results",
        "fusion_pool": "Yes",
        "marker": "s",
    },
    "self_decomposed_atomic": {
        "display": "Self-decomposed atomic prediction",
        "short": "Self-decomp. atomic",
        "file_prefix": "self_decompose",
        "result_key": "self_decompose_results",
        "fusion_pool": "Yes",
        "marker": "^",
    },
    "independent_atomic": {
        "display": "Independent atomic prediction",
        "short": "Independent atomic",
        "file_prefix": "atomic",
        "result_key": None,
        "fusion_pool": "No",
        "marker": "X",
    },
}

METHOD_ORDER = [
    "full_hypothesis",
    "atomic_prediction",
    "self_decomposed_atomic",
    "independent_atomic",
]

# Retained methods used for the atom-count analysis.
# Independent atomic prediction is kept in the candidate-level ablation,
# but it is excluded from the retained AVE prediction-pool analysis.
ATOM_BUCKET_METHOD_ORDER = [
    "full_hypothesis",
    "atomic_prediction",
    "self_decomposed_atomic",
    "independent_atomic",
]

ATOM_BUCKET_METHOD_LABELS = {
    "full_hypothesis": "Full-hypothesis",
    "atomic_prediction": "Atomic prediction",
    "self_decomposed_atomic": "Self-decomposed atomic",
    "independent_atomic": "Independent atomic",
}

# Match the line-colour order used in the AVE-LS training-summary line plots.
# The omitted second colour from that four-line plot is orange.
ATOM_BUCKET_METHOD_COLORS = {
    "full_hypothesis": "#1f77b4",
    "atomic_prediction": "#2ca02c",
    "self_decomposed_atomic": "#d62728",
    "independent_atomic": "#ff7f0e",
}

COMBO_ORDER = [
    ("qwen", "simple"),
    ("qwen", "structured"),
    ("internvl", "simple"),
    ("internvl", "structured"),
]

COMBO_LABELS = {
    ("qwen", "simple"): "Qwen simple",
    ("qwen", "structured"): "Qwen structured",
    ("internvl", "simple"): "InternVL simple",
    ("internvl", "structured"): "InternVL structured",
}

LABEL_MODE_DISPLAY = {
    "generated": "Generated label",
    "score_based": "Score-based label",
}

# Consistent plot colours. These are deliberately simple and thesis-friendly.
BLUE = "#1f77b4"
WINE = "#FF5733"
GREY = "#666666"
COMBO_COLORS = {
    ("qwen", "simple"): "#1f77b4",
    ("qwen", "structured"): "#4c9ed9",
    ("internvl", "simple"): "#8B1E3F",
    ("internvl", "structured"): "#b24b65",
}
LABEL_MODE_COLORS = {
    "generated": BLUE,
    "score_based": WINE,
}

SEP = "=" * 120
DASH = "-" * 120


# ============================================================
# 2. ARGUMENTS + PATHS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_dir",
        default=os.environ.get("DATA_ROOT", "."),
        help="Base thesis directory.",
    )
    return parser.parse_args()


def build_paths(base_dir: str) -> Dict[str, Path]:
    base = Path(base_dir)
    output_root = base / "Output" / "candidate_prediction_analysis"

    paths = {
        "base_dir": base,
        "output_root": output_root,
        "prediction_method_dir": output_root / "prediction_method_comparison",
        "label_derivation_dir": output_root / "candidate_label_derivation_stability",
        "atomicity_dir": output_root / "atomicity_analysis",
        "label_behavior_dir": output_root / "label_level_behavior",
        "report_txt": output_root / "candidate_prediction_analysis_report.txt",
    }

    for key, path in paths.items():
        if key.endswith("dir") or key == "output_root":
            path.mkdir(parents=True, exist_ok=True)

    return paths


def split_dataset_dir(base_dir: Path, split: str) -> Path:
    return base_dir / "Output" / f"{split}_dataset"


def prediction_file_path(base_dir: Path, split: str, vlm_key: str, method_key: str, prompt: str) -> Path:
    dataset_dir = split_dataset_dir(base_dir, split)
    vlm_folder = VLM_CONFIGS[vlm_key]["folder"]
    prefix = METHOD_CONFIGS[method_key]["file_prefix"]
    filename = f"{prefix}_{prompt}_{split}.jsonl"
    return dataset_dir / vlm_folder / filename


def reference_decomposition_path(base_dir: Path, split: str) -> Path:
    return split_dataset_dir(base_dir, split) / f"decompose_atoms_qwen32_{split}.jsonl"


# ============================================================
# 3. BASIC HELPERS
# ============================================================

def safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(v).strip() for v in x if str(v).strip())
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()


def normalize_label(x: Any) -> str:
    text = safe_text(x).lower().strip()

    if text in LABELS:
        return text
    if "entailment" in text or "entailed" in text or "support" in text or "supported" in text:
        return "entailment"
    if "contradiction" in text or "contradict" in text or "conflict" in text or "incompatible" in text:
        return "contradiction"
    if "neutral" in text or "uncertain" in text or "unsupported" in text or "insufficient" in text or "unclear" in text:
        return "neutral"
    return "neutral"


def make_key_from_values(img_id: Any, hypo: Any, gold: Any) -> Tuple[str, str, str]:
    return (safe_text(img_id), safe_text(hypo), normalize_label(gold))


def make_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return make_key_from_values(
        row.get("Flickr30K_ID", ""),
        row.get("hypothesis", row.get("sentence2", "")),
        row.get("annotator_label", row.get("gold", row.get("label", ""))),
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_atom_texts(atoms: Any) -> List[str]:
    if not isinstance(atoms, list):
        return []

    out = []
    for atom in atoms:
        if isinstance(atom, str):
            text = safe_text(atom)
        elif isinstance(atom, dict):
            text = safe_text(
                atom.get(
                    "atom_text",
                    atom.get("atom", atom.get("claim", atom.get("fact", atom.get("text", "")))),
                )
            )
        else:
            text = ""
        if text:
            out.append(text)
    return out


def atom_bucket(num_atoms: int) -> str:
    if num_atoms <= 1:
        return "1 atom"
    if num_atoms == 2:
        return "2 atoms"
    if num_atoms == 3:
        return "3 atoms"
    return "4+ atoms"


def scores_to_probs(scores: Any) -> Dict[str, float]:
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


def argmax_label(scores: Dict[str, float]) -> str:
    return max(scores, key=scores.get)


def aggregate_independent_labels(atom_labels: List[str]) -> str:
    """
    Strict NLI aggregation for independent atom generated labels.

    Rule:
      - if any atom is contradiction -> contradiction
      - else if all atoms are entailment -> entailment
      - else -> neutral
    """
    labels = [normalize_label(x) for x in atom_labels if safe_text(x)]
    if not labels:
        return "neutral"
    if any(label == "contradiction" for label in labels):
        return "contradiction"
    if all(label == "entailment" for label in labels):
        return "entailment"
    return "neutral"


def aggregate_independent_scores(atom_scores: List[Dict[str, float]]) -> str:
    """
    Score-sum aggregation for independent atomic prediction.

    Each atom has a label-likelihood score vector. We sum the scores for
    entailment, neutral, and contradiction across atoms, and return the label
    with the largest summed score.
    """
    if not atom_scores:
        return "neutral"

    summed = {lab: 0.0 for lab in LABELS}
    for scores in atom_scores:
        probs = scores_to_probs(scores)
        for lab in LABELS:
            summed[lab] += float(probs.get(lab, 0.0))

    return argmax_label(summed)


def metric_report(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    y_true = [normalize_label(x) for x in y_true]
    y_pred = [normalize_label(x) for x in y_pred]

    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)) if y_true else 0.0,
        "per_class": {
            LABELS[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in range(len(LABELS))
        },
        "confusion_matrix": cm,
        "prediction_distribution": dict(Counter(y_pred)),
        "gold_distribution": dict(Counter(y_true)),
    }


def save_plot(fig, png_path: Path, pdf_path: Optional[Path] = None) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def fmt_metric(x: float) -> str:
    return f"{x:.3f}"


# ============================================================
# 4. DATA LOADING AND PARSING
# ============================================================

def load_reference_atom_buckets(base_dir: Path, split: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    path = reference_decomposition_path(base_dir, split)
    if not path.exists():
        print(f"WARNING: reference decomposition file not found: {path}")
        return {}

    out = {}
    rows = read_jsonl(path)
    for row in rows:
        key = make_key(row)
        hypo = safe_text(row.get("hypothesis", row.get("sentence2", "")))
        atoms = ensure_atom_texts(row.get("atomic_facts", row.get("raw_atoms", [])))
        if not atoms:
            atoms = [hypo]
        out[key] = {
            "num_atoms": len(atoms),
            "atom_bucket": atom_bucket(len(atoms)),
            "atomic_facts": atoms,
        }
    return out


def parse_standard_result(row: Dict[str, Any], result_key: str) -> Tuple[str, str]:
    res = row.get(result_key, {}) or {}
    if not isinstance(res, dict):
        res = {}

    scores = scores_to_probs(
        res.get("scores")
        or res.get("probabilities")
        or res.get("class_scores")
        or res.get("score_dict")
        or {}
    )

    generated = normalize_label(
        res.get(
            "prediction",
            res.get("label", res.get("final_label", res.get("answer", res.get("model_prediction", "")))),
        )
    )
    score_based = normalize_label(
        res.get("score_prediction", res.get("score_label", res.get("score_pred", argmax_label(scores))))
    )
    return generated, score_based


def parse_independent_atomic_result(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    Parse independent atomic prediction.

    The generated-label output is derived with strict atom-label aggregation.
    The score-based output is derived by summing atom-level label-likelihood
    scores per class and taking the class with the largest summed score.
    """
    atoms = row.get("atomic_facts", [])
    if not isinstance(atoms, list):
        atoms = []

    generated_atom_labels = []
    atom_score_vectors = []

    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        res = atom.get("initial_results", {}) or {}
        if not isinstance(res, dict):
            res = {}

        scores = scores_to_probs(
            res.get("scores")
            or res.get("probabilities")
            or res.get("class_scores")
            or res.get("score_dict")
            or {}
        )

        generated_atom_labels.append(
            normalize_label(
                res.get("initial_prediction", res.get("prediction", res.get("label", res.get("final_label", ""))))
            )
        )
        atom_score_vectors.append(scores)

    generated = aggregate_independent_labels(generated_atom_labels)
    score_based = aggregate_independent_scores(atom_score_vectors)
    return generated, score_based


def load_configuration_records(
    base_dir: Path,
    split: str,
    vlm_key: str,
    method_key: str,
    prompt: str,
    ref_atoms: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> pd.DataFrame:
    path = prediction_file_path(base_dir, split, vlm_key, method_key, prompt)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")

    rows = read_jsonl(path)
    parsed_rows = []

    method_cfg = METHOD_CONFIGS[method_key]
    vlm_cfg = VLM_CONFIGS[vlm_key]

    for row_id, row in enumerate(rows):
        key = make_key(row)
        gold = normalize_label(row.get("annotator_label", row.get("gold", row.get("label", ""))))
        hypothesis = safe_text(row.get("hypothesis", row.get("sentence2", "")))
        image_id = safe_text(row.get("Flickr30K_ID", ""))

        if method_key == "independent_atomic":
            generated, score_based = parse_independent_atomic_result(row)
        else:
            generated, score_based = parse_standard_result(row, method_cfg["result_key"])

        if key in ref_atoms:
            num_atoms = ref_atoms[key]["num_atoms"]
            bucket = ref_atoms[key]["atom_bucket"]
        else:
            atoms = ensure_atom_texts(row.get("atomic_facts", []))
            if not atoms:
                atoms = [hypothesis]
            num_atoms = len(atoms)
            bucket = atom_bucket(num_atoms)

        parsed_rows.append({
            "split": split,
            "row_id": row_id,
            "instance_key": "|||".join(key),
            "Flickr30K_ID": image_id,
            "hypothesis": hypothesis,
            "gold": gold,
            "method_key": method_key,
            "method_family": method_cfg["display"],
            "method_family_short": method_cfg["short"],
            "vlm_key": vlm_key,
            "vlm": vlm_cfg["display"],
            "vlm_short": vlm_cfg["short"],
            "prompt": prompt,
            "combo_label": COMBO_LABELS[(vlm_key, prompt)],
            "generated_prediction": generated,
            "score_based_prediction": score_based,
            "num_atoms": num_atoms,
            "atom_bucket": bucket,
            "source_file": str(path),
        })

    return pd.DataFrame(parsed_rows)


def load_all_predictions(base_dir: Path) -> pd.DataFrame:
    frames = []

    for split in SPLITS:
        ref_atoms = load_reference_atom_buckets(base_dir, split)
        for vlm_key in VLM_CONFIGS:
            for method_key in METHOD_ORDER:
                for prompt in PROMPTS:
                    df = load_configuration_records(
                        base_dir=base_dir,
                        split=split,
                        vlm_key=vlm_key,
                        method_key=method_key,
                        prompt=prompt,
                        ref_atoms=ref_atoms,
                    )
                    frames.append(df)
                    print(
                        f"Loaded {split:<4} | {vlm_key:<8} | {method_key:<26} | {prompt:<10} | rows={len(df):>6}"
                    )

    all_df = pd.concat(frames, ignore_index=True)
    return all_df


# ============================================================
# 5. METRIC TABLES
# ============================================================

def build_long_metric_table(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["split", "method_key", "method_family", "vlm_key", "vlm", "vlm_short", "prompt", "combo_label"]

    for keys, sub in all_df.groupby(group_cols, sort=False):
        key_dict = dict(zip(group_cols, keys))
        y_true = sub["gold"].tolist()

        for label_mode, pred_col in [
            ("generated", "generated_prediction"),
            ("score_based", "score_based_prediction"),
        ]:
            y_pred = sub[pred_col].tolist()
            res = metric_report(y_true, y_pred)
            row = {
                **key_dict,
                "label_mode": label_mode,
                "label_mode_display": LABEL_MODE_DISPLAY[label_mode],
                "n": res["n"],
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "entailment_precision": res["per_class"]["entailment"]["precision"],
                "entailment_recall": res["per_class"]["entailment"]["recall"],
                "entailment_f1": res["per_class"]["entailment"]["f1"],
                "neutral_precision": res["per_class"]["neutral"]["precision"],
                "neutral_recall": res["per_class"]["neutral"]["recall"],
                "neutral_f1": res["per_class"]["neutral"]["f1"],
                "contradiction_precision": res["per_class"]["contradiction"]["precision"],
                "contradiction_recall": res["per_class"]["contradiction"]["recall"],
                "contradiction_f1": res["per_class"]["contradiction"]["f1"],
            }
            rows.append(row)

    return pd.DataFrame(rows)


def build_base_config_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per method x VLM x prompt = 16 rows.
    Dev/test generated and score-based metrics are placed as columns.
    """
    id_cols = ["method_key", "method_family", "vlm_key", "vlm", "vlm_short", "prompt", "combo_label"]
    rows = []

    base = long_df[id_cols].drop_duplicates().reset_index(drop=True)

    for _, base_row in base.iterrows():
        row = base_row.to_dict()
        sub_base = long_df[
            (long_df["method_key"] == row["method_key"])
            & (long_df["vlm_key"] == row["vlm_key"])
            & (long_df["prompt"] == row["prompt"])
        ]

        for split in SPLITS:
            sub_split = sub_base[sub_base["split"] == split]
            for mode in ["generated", "score_based"]:
                sub_mode = sub_split[sub_split["label_mode"] == mode]
                if sub_mode.empty:
                    row[f"{split}_{mode}_accuracy"] = np.nan
                    row[f"{split}_{mode}_macro_f1"] = np.nan
                else:
                    r = sub_mode.iloc[0]
                    row[f"{split}_{mode}_accuracy"] = float(r["accuracy"])
                    row[f"{split}_{mode}_macro_f1"] = float(r["macro_f1"])

            # Best mode for this split, based on accuracy then macro-F1.
            if not sub_split.empty:
                best = sub_split.sort_values(["accuracy", "macro_f1"], ascending=[False, False]).iloc[0]
                row[f"{split}_best_label_mode"] = best["label_mode"]
                row[f"{split}_best_label_mode_display"] = best["label_mode_display"]
                row[f"{split}_best_accuracy"] = float(best["accuracy"])
                row[f"{split}_best_macro_f1"] = float(best["macro_f1"])
            else:
                row[f"{split}_best_label_mode"] = ""
                row[f"{split}_best_label_mode_display"] = ""
                row[f"{split}_best_accuracy"] = np.nan
                row[f"{split}_best_macro_f1"] = np.nan

        rows.append(row)

    wide = pd.DataFrame(rows)

    method_order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    combo_order_map = {c: i for i, c in enumerate(COMBO_ORDER)}
    wide["_method_order"] = wide["method_key"].map(method_order_map)
    wide["_combo_order"] = wide.apply(lambda r: combo_order_map[(r["vlm_key"], r["prompt"])], axis=1)
    wide = wide.sort_values(["_method_order", "_combo_order"]).drop(columns=["_method_order", "_combo_order"])
    return wide.reset_index(drop=True)


def build_method_family_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact main-text table.

    For each method family, select the best base configuration on the
    development set. The same selected configuration and same label-decision
    rule are then evaluated on the test set. This avoids selecting a separate
    best configuration on the test split.
    """
    rows = []

    for method_key in METHOD_ORDER:
        method_rows = wide_df[wide_df["method_key"] == method_key].copy()
        if method_rows.empty:
            continue

        # Select once on development accuracy, using macro-F1 as tie-breaker.
        best_dev = method_rows.sort_values(
            ["dev_best_accuracy", "dev_best_macro_f1"],
            ascending=[False, False],
        ).iloc[0]

        selected_mode = best_dev["dev_best_label_mode"]
        selected_mode_display = best_dev["dev_best_label_mode_display"]
        selected_config = f"{best_dev['vlm_short']} {best_dev['prompt']} ({selected_mode_display})"

        row = {
            "method_key": method_key,
            "method_family": METHOD_CONFIGS[method_key]["display"],
            "fusion_pool": METHOD_CONFIGS[method_key]["fusion_pool"],
            "selected_vlm": best_dev["vlm_short"],
            "selected_prompt": best_dev["prompt"],
            "selected_label_mode": selected_mode,
            "selected_label_mode_display": selected_mode_display,
            "selected_configuration": selected_config,

            # Keep these names for compatibility with the report/export code.
            "dev_best_configuration": selected_config,
            "test_best_configuration": selected_config,

            # Development-selected configuration evaluated on development.
            "dev_best_accuracy": float(best_dev[f"dev_{selected_mode}_accuracy"]),
            "dev_best_macro_f1": float(best_dev[f"dev_{selected_mode}_macro_f1"]),

            # Same development-selected configuration evaluated on test.
            "test_best_accuracy": float(best_dev[f"test_{selected_mode}_accuracy"]),
            "test_best_macro_f1": float(best_dev[f"test_{selected_mode}_macro_f1"]),

            # Compact indication of how much the four VLM/prompt candidates vary.
            # Each base configuration contributes its best label-decision rule
            # on that split.
            "dev_accuracy_range": float(
                method_rows["dev_best_accuracy"].max() - method_rows["dev_best_accuracy"].min()
            ),
            "test_accuracy_range": float(
                method_rows["test_best_accuracy"].max() - method_rows["test_best_accuracy"].min()
            ),
            "dev_macro_f1_range": float(
                method_rows["dev_best_macro_f1"].max() - method_rows["dev_best_macro_f1"].min()
            ),
            "test_macro_f1_range": float(
                method_rows["test_best_macro_f1"].max() - method_rows["test_best_macro_f1"].min()
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def build_label_derivation_stability_table(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per base configuration x label derivation mode = 32 rows.
    """
    rows = []

    for _, row in wide_df.iterrows():
        for mode in ["generated", "score_based"]:
            rows.append({
                "method_key": row["method_key"],
                "method_family": row["method_family"],
                "vlm_key": row["vlm_key"],
                "vlm": row["vlm"],
                "vlm_short": row["vlm_short"],
                "prompt": row["prompt"],
                "combo_label": row["combo_label"],
                "label_mode": mode,
                "label_mode_display": LABEL_MODE_DISPLAY[mode],
                "dev_accuracy": float(row[f"dev_{mode}_accuracy"]),
                "dev_macro_f1": float(row[f"dev_{mode}_macro_f1"]),
                "test_accuracy": float(row[f"test_{mode}_accuracy"]),
                "test_macro_f1": float(row[f"test_{mode}_macro_f1"]),
                "test_minus_dev_accuracy": float(row[f"test_{mode}_accuracy"] - row[f"dev_{mode}_accuracy"]),
                "test_minus_dev_macro_f1": float(row[f"test_{mode}_macro_f1"] - row[f"dev_{mode}_macro_f1"]),
            })

    stability = pd.DataFrame(rows)
    method_order_map = {m: i for i, m in enumerate(METHOD_ORDER)}
    stability["_method_order"] = stability["method_key"].map(method_order_map)
    stability = stability.sort_values(["_method_order", "vlm_key", "prompt", "label_mode"]).drop(columns=["_method_order"])
    return stability.reset_index(drop=True)


def build_label_derivation_summary(stability_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for method_key in METHOD_ORDER:
        sub_method = stability_df[stability_df["method_key"] == method_key]
        row = {
            "method_key": method_key,
            "method_family": METHOD_CONFIGS[method_key]["display"],
        }
        for mode in ["generated", "score_based"]:
            sub = sub_method[sub_method["label_mode"] == mode]
            vals = sub[["dev_accuracy", "test_accuracy"]].to_numpy().reshape(-1)
            row[f"{mode}_mean_accuracy"] = float(np.mean(vals)) if len(vals) else np.nan
            row[f"{mode}_std_accuracy"] = float(np.std(vals)) if len(vals) else np.nan
            row[f"{mode}_min_accuracy"] = float(np.min(vals)) if len(vals) else np.nan
            row[f"{mode}_max_accuracy"] = float(np.max(vals)) if len(vals) else np.nan
            row[f"{mode}_range_accuracy"] = float(np.max(vals) - np.min(vals)) if len(vals) else np.nan
        row["better_mean_rule"] = (
            "Generated label"
            if row["generated_mean_accuracy"] >= row["score_based_mean_accuracy"]
            else "Score-based label"
        )
        row["more_stable_rule"] = (
            "Generated label"
            if row["generated_range_accuracy"] <= row["score_based_range_accuracy"]
            else "Score-based label"
        )
        rows.append(row)

    return pd.DataFrame(rows)

def build_prediction_method_divergence_table(
    all_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the divergence counts between the best full-hypothesis configuration
    and the best configuration of each atomic method family.

    For each split and each compared method family, compute:
      - FH correct, method wrong
      - Method correct, FH wrong
      - Net gain = (method correct, FH wrong) - (FH correct, method wrong)

    The best configuration for each method family is selected on the development set.
    """
    fh_config = choose_dev_best_config(wide_df, ["full_hypothesis"])

    comparison_method_keys = [
        "atomic_prediction",
        "self_decomposed_atomic",
        "independent_atomic",
    ]

    rows = []

    for method_key in comparison_method_keys:
        method_config = choose_dev_best_config(wide_df, [method_key])

        for split in SPLITS:
            fh_df = get_predictions_for_config(
                all_df=all_df,
                split=split,
                method_key=fh_config["method_key"],
                vlm_key=fh_config["vlm_key"],
                prompt=fh_config["prompt"],
                label_mode=fh_config["dev_best_label_mode"],
            )[["instance_key", "gold", "prediction"]].rename(
                columns={"prediction": "fh_prediction"}
            )

            method_df = get_predictions_for_config(
                all_df=all_df,
                split=split,
                method_key=method_config["method_key"],
                vlm_key=method_config["vlm_key"],
                prompt=method_config["prompt"],
                label_mode=method_config["dev_best_label_mode"],
            )[["instance_key", "gold", "prediction"]].rename(
                columns={"prediction": "method_prediction"}
            )

            merged = fh_df.merge(
                method_df[["instance_key", "method_prediction"]],
                on="instance_key",
                how="inner",
            )

            fh_correct = merged["fh_prediction"] == merged["gold"]
            method_correct = merged["method_prediction"] == merged["gold"]

            fh_correct_method_wrong = int((fh_correct & ~method_correct).sum())
            method_correct_fh_wrong = int((method_correct & ~fh_correct).sum())
            both_correct = int((fh_correct & method_correct).sum())
            both_wrong = int((~fh_correct & ~method_correct).sum())
            net_gain = method_correct_fh_wrong - fh_correct_method_wrong

            rows.append({
                "split": split,
                "comparison_method_key": method_key,
                "comparison_method_family": METHOD_CONFIGS[method_key]["display"],
                "comparison_method_short": METHOD_CONFIGS[method_key]["short"],

                "fh_vlm_short": fh_config["vlm_short"],
                "fh_prompt": fh_config["prompt"],
                "fh_label_mode": fh_config["dev_best_label_mode"],
                "fh_label_mode_display": LABEL_MODE_DISPLAY[fh_config["dev_best_label_mode"]],

                "method_vlm_short": method_config["vlm_short"],
                "method_prompt": method_config["prompt"],
                "method_label_mode": method_config["dev_best_label_mode"],
                "method_label_mode_display": LABEL_MODE_DISPLAY[method_config["dev_best_label_mode"]],

                "n_common": int(len(merged)),
                "fh_correct_method_wrong": fh_correct_method_wrong,
                "method_correct_fh_wrong": method_correct_fh_wrong,
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "net_gain": net_gain,
            })

    return pd.DataFrame(rows)


def plot_prediction_method_divergence(div_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Divergence plot comparing the best full-hypothesis configuration against the
    best configuration of each atomic method family.

    Left (pink):  full-hypothesis correct, atomic method wrong
    Right (green): atomic method correct, full-hypothesis wrong
    """
    pink = "#e85d9e"
    green = "#7ecb4f"

    comparison_order = [
        "independent_atomic",
        "atomic_prediction",
        "self_decomposed_atomic",
    ]

    method_labels = {
        "independent_atomic": "Independent atomic",
        "atomic_prediction": "Atomic prediction",
        "self_decomposed_atomic": "Self-decomposed atomic",
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.4), sharey=True)

    panels = [
        (axes[0], "dev", "Development"),
        (axes[1], "test", "Test"),
    ]

    max_abs = 0
    for _, split, _ in panels:
        sub = div_df[div_df["split"] == split]
        if not sub.empty:
            max_abs = max(
                max_abs,
                int(sub["fh_correct_method_wrong"].max()),
                int(sub["method_correct_fh_wrong"].max()),
            )
    max_abs = max(max_abs, 1)

    for ax, split, title in panels:
        sub = div_df[div_df["split"] == split].copy()
        sub["order"] = sub["comparison_method_key"].map(
            {k: i for i, k in enumerate(comparison_order)}
        )
        sub = sub.sort_values("order")

        y = np.arange(len(sub))
        bar_height = 0.42

        left_vals = -sub["fh_correct_method_wrong"].values
        right_vals = sub["method_correct_fh_wrong"].values
        net_vals = sub["net_gain"].values

        ax.barh(y, left_vals, color=pink, edgecolor="none", height=bar_height)
        ax.barh(y, right_vals, color=green, edgecolor="none", height=bar_height)

        ax.axvline(0, color="#444444", linewidth=1.4)

        ax.set_title(title, fontsize=14, pad=6, fontweight="normal")
        ax.set_xlabel("Instance count", fontsize=12)

        ax.set_yticks(y)
        ax.set_yticklabels(
            [method_labels[k] for k in sub["comparison_method_key"]],
            fontsize=11,
        )

        ax.tick_params(axis="x", labelsize=11)
        ax.grid(axis="x", alpha=0.2, linewidth=0.8)
        ax.set_axisbelow(True)

        ax.set_ylim(len(sub) - 0.55, -0.55)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(int(x))}"))

        x_right_margin = max_abs * 0.55
        x_left_margin = max_abs * 0.15
        ax.set_xlim(-(max_abs + x_left_margin), max_abs + x_right_margin)

        for yi, lv, rv, nv in zip(y, left_vals, right_vals, net_vals):
            left_count = abs(int(lv))
            right_count = int(rv)
            net = int(nv)

            if left_count > 0:
                ax.text(
                    lv / 2, yi,
                    f"{left_count}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if left_count >= 80 else "black",
                )

            if right_count > 0:
                ax.text(
                    rv / 2, yi,
                    f"{right_count}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if right_count >= 80 else "black",
                )

            net_color = "#000000"
            ax.text(
                max_abs + x_right_margin * 0.08, yi,
                f"net {net:+d}",
                ha="left", va="center",
                fontsize=11,
                color=net_color,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        mpatches.Patch(color=pink,
                       label="Full-hypothesis correct, atomic method wrong"),
        mpatches.Patch(color=green,
                       label="Atomic method correct, full-hypothesis wrong"),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(
        out_dir / "full_hypothesis_vs_atomic_divergence_dev_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "full_hypothesis_vs_atomic_divergence_dev_test.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# 6. PREDICTION METHOD COMPARISON OUTPUTS
# ============================================================

def export_prediction_method_outputs(
    all_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    paths: Dict[str, Path],
) -> None:
    """
    Export the human-readable tables for the individual candidate-performance
    analysis and add the divergence plot comparing FH against the atomic
    method families.

    Outputs:
      1. method_family_summary_main_table.txt
      2. full_16_configuration_results.txt
      3. full_hypothesis_vs_atomic_divergence_summary.txt
      4. full_hypothesis_vs_atomic_divergence_summary.csv
      5. full_hypothesis_vs_atomic_divergence_dev_test.png/.pdf
    """
    out_dir = paths["prediction_method_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale artifacts from earlier versions
    stale_files = [
        "candidate_method_comparison.png",
        "candidate_method_comparison.pdf",
        "method_family_summary.csv",
        "method_family_summary_main_table.csv",
        "full_16_configuration_results.csv",
        "method_family_summary_main_table_latex.tex",
        "full_16_configuration_results_latex.tex",
    ]
    for fname in stale_files:
        stale_path = out_dir / fname
        if stale_path.exists():
            stale_path.unlink()

    # ------------------------------------------------------------------
    # Compact main-text table
    # ------------------------------------------------------------------
    main_cols = [
        "method_family",
        "selected_configuration",
        "dev_best_accuracy",
        "dev_best_macro_f1",
        "test_best_accuracy",
        "test_best_macro_f1",
        "dev_accuracy_range",
        "test_accuracy_range",
        "fusion_pool",
    ]
    main_table = summary_df[main_cols].copy()

    with open(out_dir / "method_family_summary_main_table.txt", "w", encoding="utf-8") as f:
        f.write("Compact table: best development-selected configuration per method family\n")
        f.write("=" * 120 + "\n")
        f.write(main_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        f.write("\n")

    # ------------------------------------------------------------------
    # Full 16-row table
    # ------------------------------------------------------------------
    full_cols = [
        "method_family",
        "vlm_short",
        "prompt",
        "dev_generated_accuracy",
        "dev_generated_macro_f1",
        "dev_score_based_accuracy",
        "dev_score_based_macro_f1",
        "test_generated_accuracy",
        "test_generated_macro_f1",
        "test_score_based_accuracy",
        "test_score_based_macro_f1",
    ]
    full_table = wide_df[full_cols].copy()

    with open(out_dir / "full_16_configuration_results.txt", "w", encoding="utf-8") as f:
        f.write("Full 16-row table: method family x VLM x prompt configurations\n")
        f.write("=" * 140 + "\n")
        f.write(full_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        f.write("\n")

    # ------------------------------------------------------------------
    # Divergence summary + plot
    # ------------------------------------------------------------------
    div_df = build_prediction_method_divergence_table(all_df, wide_df)
    div_df.to_csv(out_dir / "full_hypothesis_vs_atomic_divergence_summary.csv", index=False)

    txt_cols = [
        "split",
        "comparison_method_family",
        "fh_vlm_short",
        "fh_prompt",
        "fh_label_mode_display",
        "method_vlm_short",
        "method_prompt",
        "method_label_mode_display",
        "n_common",
        "fh_correct_method_wrong",
        "method_correct_fh_wrong",
        "net_gain",
        "both_correct",
        "both_wrong",
    ]

    with open(out_dir / "full_hypothesis_vs_atomic_divergence_summary.txt", "w", encoding="utf-8") as f:
        f.write("Divergence summary: full-hypothesis versus atomic method families\n")
        f.write("=" * 140 + "\n")
        f.write(div_df[txt_cols].to_string(index=False))
        f.write("\n")

    plot_prediction_method_divergence(div_df, out_dir)

# ============================================================
# 7. LABEL DERIVATION STABILITY OUTPUTS
# ============================================================

def plot_label_derivation_stability(stability_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Development-test stability scatter plot, split by VLM.

    Each point is one:
        method family x prompt x label-derivation rule

    x-axis: development accuracy
    y-axis: test accuracy
    """

    plot_df = stability_df.copy()

    # Small deterministic offsets reduce overplotting while preserving the pattern.
    combo_rank_map = {combo: idx for idx, combo in enumerate(COMBO_ORDER)}
    mode_offset = {"generated": -1, "score_based": 1}

    def _stable_offsets(row):
        combo_idx = combo_rank_map[(row["vlm_key"], row["prompt"])]
        method_idx = METHOD_ORDER.index(row["method_key"])
        mode_idx = mode_offset[row["label_mode"]]
        seed = (combo_idx * 17) + (method_idx * 5) + mode_idx
        dx = (seed % 7 - 3) * 0.0012
        dy = ((seed // 2) % 7 - 3) * 0.0012
        return pd.Series({"jx": dx, "jy": dy})

    plot_df[["jx", "jy"]] = plot_df.apply(_stable_offsets, axis=1)
    plot_df["dev_plot"] = plot_df["dev_accuracy"] + plot_df["jx"]
    plot_df["test_plot"] = plot_df["test_accuracy"] + plot_df["jy"]

    # Global limits for both panels
    vals = np.concatenate([
        plot_df["dev_accuracy"].values,
        plot_df["test_accuracy"].values,
    ])
    min_v = max(0.0, float(np.nanmin(vals)) - 0.02)
    max_v = min(1.0, float(np.nanmax(vals)) + 0.02)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), sharex=True, sharey=True)

    panels = [
        (axes[0], "qwen", "Qwen3-VL-8B"),
        (axes[1], "internvl", "InternVL3-8B"),
    ]

    def _plot_points(ax, df, marker_size=165, alpha=0.84):
        for method_key in METHOD_ORDER:
            for mode in ["generated", "score_based"]:
                sub = df[
                    (df["method_key"] == method_key)
                    & (df["label_mode"] == mode)
                ]
                if sub.empty:
                    continue

                ax.scatter(
                    sub["dev_plot"],
                    sub["test_plot"],
                    s=marker_size,
                    marker=METHOD_CONFIGS[method_key]["marker"],
                    color=LABEL_MODE_COLORS[mode],
                    edgecolor="black",
                    linewidth=0.75,
                    alpha=alpha,
                    zorder=3,
                )

    for idx, (ax, vlm_key, title) in enumerate(panels):
        vlm_df = plot_df[plot_df["vlm_key"] == vlm_key].copy()

        _plot_points(ax, vlm_df, marker_size=165, alpha=0.84)

        # Diagonal line: equal dev and test accuracy
        ax.plot(
            [min_v, max_v],
            [min_v, max_v],
            linestyle="--",
            color=GREY,
            linewidth=1.3,
            alpha=0.9,
            zorder=2,
        )

        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        ax.set_title(title, fontsize=16, pad=8)
        ax.set_xlabel("Development accuracy", fontsize=15, labelpad=7)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.24, linestyle="-", linewidth=0.8)
        ax.set_axisbelow(True)

        if idx == 0:
            ax.set_ylabel("Test accuracy", fontsize=15, labelpad=7)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

    # ------------------------------------------------------------
    # Inset only for Qwen panel
    # ------------------------------------------------------------
    qwen_df = plot_df[plot_df["vlm_key"] == "qwen"].copy()
    ax_left = axes[0]

    if not qwen_df.empty:
        qwen_vals = np.concatenate([
            qwen_df["dev_accuracy"].values,
            qwen_df["test_accuracy"].values,
        ])

        zoom_min = max(0.0, float(np.nanmin(qwen_vals)) - 0.008)
        zoom_max = min(1.0, float(np.nanmax(qwen_vals)) + 0.008)

        ax_inset = ax_left.inset_axes([0.51, 0.07, 0.43, 0.43])

        _plot_points(ax_inset, qwen_df, marker_size=95, alpha=0.88)

        ax_inset.plot(
            [zoom_min, zoom_max],
            [zoom_min, zoom_max],
            linestyle="--",
            color=GREY,
            linewidth=1.0,
            alpha=0.9,
            zorder=2,
        )

        ax_inset.set_xlim(zoom_min, zoom_max)
        ax_inset.set_ylim(zoom_min, zoom_max)
        ax_inset.grid(alpha=0.30, linestyle="-", linewidth=0.6)
        ax_inset.set_axisbelow(True)
        ax_inset.tick_params(labelsize=8)
        ax_inset.set_title("Zoomed cluster", fontsize=11, pad=3)

        ax_left.indicate_inset_zoom(
            ax_inset,
            edgecolor="#555555",
            alpha=0.25,
            linewidth=0.8,
        )

    # ------------------------------------------------------------
    # Legends: BOTH legends in BOTH panels
    # ------------------------------------------------------------
    shape_handles = [
        Line2D(
            [0], [0],
            marker=METHOD_CONFIGS[m]["marker"],
            color="w",
            label=METHOD_CONFIGS[m]["short"],
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=8.5,
        )
        for m in METHOD_ORDER
    ]

    color_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=LABEL_MODE_DISPLAY[m],
            markerfacecolor=LABEL_MODE_COLORS[m],
            markeredgecolor="black",
            markersize=8.5,
        )
        for m in ["generated", "score_based"]
    ]

    for ax in axes:
        leg1 = ax.legend(
            handles=shape_handles,
            title="Method family",
            loc="upper left",
            fontsize=10.5,
            title_fontsize=12,
            frameon=True,
            borderpad=0.6,
            labelspacing=0.4,
            handletextpad=0.5,
        )
        ax.add_artist(leg1)

        ax.legend(
            handles=color_handles,
            title="Label derivation",
            loc="upper left",
            bbox_to_anchor=(0.0, 0.70),
            fontsize=10.5,
            title_fontsize=12,
            frameon=True,
            borderpad=0.6,
            labelspacing=0.4,
            handletextpad=0.5,
        )

    fig.tight_layout(rect=[0, 0, 1, 1])

    save_plot(
        fig,
        out_dir / "candidate_label_derivation_stability_scatter_updated.png",
        out_dir / "candidate_label_derivation_stability_scatter_updated.pdf",
    )


def export_label_derivation_outputs(stability_df: pd.DataFrame, summary_df: pd.DataFrame, paths: Dict[str, Path]) -> None:
    out_dir = paths["label_derivation_dir"]
    stability_df.to_csv(out_dir / "candidate_label_derivation_stability_32_points.csv", index=False)
    summary_df.to_csv(out_dir / "candidate_label_derivation_stability_summary.csv", index=False)

    with open(out_dir / "candidate_label_derivation_stability_32_points_latex.tex", "w", encoding="utf-8") as f:
        latex_cols = [
            "method_family", "vlm_short", "prompt", "label_mode_display",
            "dev_accuracy", "dev_macro_f1", "test_accuracy", "test_macro_f1", "test_minus_dev_accuracy",
        ]
        f.write(stability_df[latex_cols].to_latex(index=False, escape=True, float_format="%.3f"))

    plot_label_derivation_stability(stability_df, out_dir)


# ============================================================
# 8. ATOMICITY ANALYSIS OUTPUTS
# ============================================================

def choose_dev_best_config(
    wide_df: pd.DataFrame,
    method_keys: List[str],
) -> Dict[str, Any]:
    sub = wide_df[wide_df["method_key"].isin(method_keys)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for method keys: {method_keys}")
    best = sub.sort_values(["dev_best_accuracy", "dev_best_macro_f1"], ascending=[False, False]).iloc[0]
    return best.to_dict()


def get_predictions_for_config(
    all_df: pd.DataFrame,
    split: str,
    method_key: str,
    vlm_key: str,
    prompt: str,
    label_mode: str,
) -> pd.DataFrame:
    sub = all_df[
        (all_df["split"] == split)
        & (all_df["method_key"] == method_key)
        & (all_df["vlm_key"] == vlm_key)
        & (all_df["prompt"] == prompt)
    ].copy()

    if "instance_key" not in sub.columns:
        sub["instance_key"] = sub.apply(
            lambda r: "|||".join(make_key_from_values(r["Flickr30K_ID"], r["hypothesis"], r["gold"])),
            axis=1,
        )

    pred_col = "generated_prediction" if label_mode == "generated" else "score_based_prediction"
    sub["prediction"] = sub[pred_col]
    return sub


def build_selected_configs_for_atom_bucket_analysis(wide_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Select one development-best configuration for each method family used in
    the atom-count analysis.

    This includes full-hypothesis prediction, the two retained atomic method
    families, and independent atomic prediction as an ablation method.
    """
    selected = []

    for method_key in ATOM_BUCKET_METHOD_ORDER:
        best = choose_dev_best_config(wide_df, [method_key])
        selected.append({
            "config_role": ATOM_BUCKET_METHOD_LABELS[method_key],
            **best,
        })

    return selected


def build_matched_atom_bucket_metrics(
    all_df: pd.DataFrame,
    selected_configs: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Compute per-instance matched accuracy by atom-count bucket.

    For each split and each atom bucket, the denominator is the common set of
    instance keys available for all selected method configurations. This makes
    the FH, AP, and SD lines directly comparable inside each bucket.
    """
    prepared: Dict[Tuple[str, str], pd.DataFrame] = {}

    for config in selected_configs:
        label_mode = config["dev_best_label_mode"]
        for split in SPLITS:
            pred_df = get_predictions_for_config(
                all_df=all_df,
                split=split,
                method_key=config["method_key"],
                vlm_key=config["vlm_key"],
                prompt=config["prompt"],
                label_mode=label_mode,
            )
            prepared[(split, config["config_role"])] = pred_df

    rows = []

    for split in SPLITS:
        for bucket in BUCKETS:
            key_sets = []
            available_counts = {}

            for config in selected_configs:
                role = config["config_role"]
                sub = prepared[(split, role)]
                sub_bucket = sub[sub["atom_bucket"] == bucket]
                keys = set(sub_bucket["instance_key"].tolist())
                key_sets.append(keys)
                available_counts[role] = len(keys)

            common_keys = set.intersection(*key_sets) if key_sets else set()

            for config in selected_configs:
                role = config["config_role"]
                label_mode = config["dev_best_label_mode"]
                sub = prepared[(split, role)]
                sub = sub[
                    (sub["atom_bucket"] == bucket)
                    & (sub["instance_key"].isin(common_keys))
                ].copy()

                if sub.empty:
                    accuracy = np.nan
                    macro_f1 = np.nan
                    n = 0
                else:
                    res = metric_report(sub["gold"].tolist(), sub["prediction"].tolist())
                    accuracy = res["accuracy"]
                    macro_f1 = res["macro_f1"]
                    n = res["n"]

                rows.append({
                    "split": split,
                    "config_role": role,
                    "method_key": config["method_key"],
                    "method_family": config["method_family"],
                    "vlm_short": config["vlm_short"],
                    "prompt": config["prompt"],
                    "label_mode": label_mode,
                    "label_mode_display": LABEL_MODE_DISPLAY[label_mode],
                    "atom_bucket": bucket,
                    "available_n": available_counts[role],
                    "matched_n": len(common_keys),
                    "n": n,
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                })

    return pd.DataFrame(rows)


def plot_atomicity_bucket_accuracy(bucket_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Line graph for atom-count analysis.

    Each line is one retained method family. Each point is the per-instance
    matched accuracy for that method on the same examples in the corresponding
    atom-count bucket.
    """
    x_positions = np.arange(len(BUCKETS))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    for ax, split, panel_title in [
        (axes[0], "dev", "Development"),
        (axes[1], "test", "Test"),
    ]:
        for method_key in ATOM_BUCKET_METHOD_ORDER:
            role = ATOM_BUCKET_METHOD_LABELS[method_key]
            sub = bucket_df[
                (bucket_df["split"] == split)
                & (bucket_df["config_role"] == role)
            ].copy()

            sub["bucket_order"] = sub["atom_bucket"].map(
                {b: i for i, b in enumerate(BUCKETS)}
            )
            sub = sub.sort_values("bucket_order")

            ax.plot(
                sub["bucket_order"].values,
                sub["accuracy"].values,
                marker="o",
                linewidth=2,
                markersize=5,
                label=role,
                color=ATOM_BUCKET_METHOD_COLORS[method_key],
            )

        ax.set_title(panel_title, fontsize=14, pad=8)
        ax.set_xlabel("Atom-count bucket", fontsize=13)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(BUCKETS, fontsize=11)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, frameon=True, loc="best")

        vals = bucket_df[bucket_df["split"] == split]["accuracy"].dropna().values
        if len(vals) > 0:
            low = max(0.0, vals.min() - 0.01)
            high = min(1.0, vals.max() + 0.01)
            ax.set_ylim(low, high)

    # Keep y-axis label and tick labels only on the left panel
    axes[0].set_ylabel("Accuracy", fontsize=13)
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", left=False, labelleft=False)

    fig.tight_layout(rect=[0, 0, 1, 0.98])

    fig.savefig(
        out_dir / "atom_bucket_accuracy_dev_test_line.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "atom_bucket_accuracy_dev_test_line.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def export_atomicity_outputs(all_df: pd.DataFrame, wide_df: pd.DataFrame, paths: Dict[str, Path]) -> pd.DataFrame:
    out_dir = paths["atomicity_dir"]

    selected_configs = build_selected_configs_for_atom_bucket_analysis(wide_df)
    selected_df = pd.DataFrame(selected_configs)
    selected_df.to_csv(out_dir / "atomicity_selected_configurations.csv", index=False)

    bucket_df = build_matched_atom_bucket_metrics(all_df, selected_configs)
    bucket_df.to_csv(out_dir / "atom_bucket_accuracy_dev_test_matched.csv", index=False)

    with open(out_dir / "atom_bucket_accuracy_dev_test_matched.txt", "w", encoding="utf-8") as f:
        table_cols = [
            "split", "config_role", "atom_bucket", "matched_n", "accuracy", "macro_f1",
            "vlm_short", "prompt", "label_mode_display",
        ]
        f.write("Per-instance matched atom-bucket accuracy table\n")
        f.write("=" * 100 + "\n")
        f.write(bucket_df[table_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        f.write("\n")

    plot_atomicity_bucket_accuracy(bucket_df, out_dir)
    return bucket_df


# ============================================================
# 9. LABEL-LEVEL BEHAVIOUR OUTPUTS
# ============================================================

def build_per_class_recall_for_best_family_configs(
    all_df: pd.DataFrame,
    wide_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    selected_rows = []
    recall_rows = []

    for method_key in METHOD_ORDER:
        best = choose_dev_best_config(wide_df, [method_key])
        selected_rows.append({"selection_role": "best_dev_config_per_method_family", **best})

        label_mode = best["dev_best_label_mode"]
        for split in SPLITS:
            pred_df = get_predictions_for_config(
                all_df=all_df,
                split=split,
                method_key=best["method_key"],
                vlm_key=best["vlm_key"],
                prompt=best["prompt"],
                label_mode=label_mode,
            )
            res = metric_report(pred_df["gold"].tolist(), pred_df["prediction"].tolist())
            recall_rows.append({
                "split": split,
                "method_key": method_key,
                "method_family": METHOD_CONFIGS[method_key]["display"],
                "method_family_short": METHOD_CONFIGS[method_key]["short"],
                "selected_vlm": best["vlm_short"],
                "selected_prompt": best["prompt"],
                "selected_label_mode": label_mode,
                "selected_label_mode_display": LABEL_MODE_DISPLAY[label_mode],
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "entailment_recall": res["per_class"]["entailment"]["recall"],
                "neutral_recall": res["per_class"]["neutral"]["recall"],
                "contradiction_recall": res["per_class"]["contradiction"]["recall"],
            })

    # Best individual standalone predictor across all method families, selected on dev.
    best_individual = choose_dev_best_config(wide_df, METHOD_ORDER)

    return pd.DataFrame(selected_rows), pd.DataFrame(recall_rows), best_individual


def build_confusion_rows_for_config(all_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    label_mode = config["dev_best_label_mode"]

    for split in SPLITS:
        pred_df = get_predictions_for_config(
            all_df=all_df,
            split=split,
            method_key=config["method_key"],
            vlm_key=config["vlm_key"],
            prompt=config["prompt"],
            label_mode=label_mode,
        )
        res = metric_report(pred_df["gold"].tolist(), pred_df["prediction"].tolist())
        cm = res["confusion_matrix"].astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

        for i, gold in enumerate(LABELS):
            for j, pred in enumerate(LABELS):
                rows.append({
                    "split": split,
                    "gold_label": gold,
                    "predicted_label": pred,
                    "count": int(cm[i, j]),
                    "row_normalized": float(cm_norm[i, j]),
                    "selected_method_key": config["method_key"],
                    "selected_method_family": config["method_family"],
                    "selected_vlm": config["vlm_short"],
                    "selected_prompt": config["prompt"],
                    "selected_label_mode": label_mode,
                    "selected_label_mode_display": LABEL_MODE_DISPLAY[label_mode],
                    "accuracy": res["accuracy"],
                    "macro_f1": res["macro_f1"],
                })

    return pd.DataFrame(rows)


def recall_matrix(recall_df: pd.DataFrame, split: str) -> np.ndarray:
    rows = []
    for method_key in METHOD_ORDER:
        sub = recall_df[(recall_df["split"] == split) & (recall_df["method_key"] == method_key)]
        if sub.empty:
            rows.append([np.nan, np.nan, np.nan])
        else:
            r = sub.iloc[0]
            rows.append([r["entailment_recall"], r["neutral_recall"], r["contradiction_recall"]])
    return np.array(rows, dtype=float)


def confusion_matrix_from_rows(conf_df: pd.DataFrame, split: str) -> np.ndarray:
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=float)
    sub = conf_df[conf_df["split"] == split]
    for _, row in sub.iterrows():
        i = LABELS.index(row["gold_label"])
        j = LABELS.index(row["predicted_label"])
        mat[i, j] = float(row["row_normalized"])
    return mat


def plot_recall_heatmaps(recall_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Per-class recall heatmap for the best development-selected configuration
    of each method family.
    """
    warm_soft = LinearSegmentedColormap.from_list(
        "warm_soft",
        ["#fffaf3", "#f6e7d4", "#ebcfb2", "#ddb58d", "#c99872"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.3), constrained_layout=True)
    display_labels = [LABEL_TO_SHORT[l] for l in LABELS]
    method_labels = [METHOD_CONFIGS[m]["short"] for m in METHOD_ORDER]

    panels = [
        (axes[0], "dev", "Development"),
        (axes[1], "test", "Test"),
    ]

    for ax, split, title in panels:
        mat = recall_matrix(recall_df, split)
        image = ax.imshow(mat, vmin=0.4, vmax=1.0, cmap=warm_soft)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Gold label")

        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(METHOD_ORDER)))
        ax.set_xticklabels(display_labels)

        if split == "dev":
            ax.set_ylabel("Method family")
            ax.set_yticklabels(method_labels)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                if np.isnan(value):
                    text = ""
                    text_color = "black"
                else:
                    text = f"{value:.3f}"
                    text_color = "white" if value >= 0.75 or value <= 0.25 else "black"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="bold" if not np.isnan(value) and value >= 0.80 else "normal",
                )

    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
    colorbar.set_label("Recall")

    fig.savefig(out_dir / "per_class_recall_heatmap_dev_test.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "per_class_recall_heatmap_dev_test.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_best_individual_confusion_matrices(conf_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Row-normalised confusion matrices for the single best individual candidate,
    selected on the development set.
    """
    dev_norm = confusion_matrix_from_rows(conf_df, "dev")
    test_norm = confusion_matrix_from_rows(conf_df, "test")

    display_labels = [LABEL_TO_SHORT[l] for l in LABELS]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    panels = [("Development", dev_norm), ("Test", test_norm)]

    for idx, (ax, (title, matrix)) in enumerate(zip(axes, panels)):
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="PuBuGn")

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Predicted label")

        # Keep y-axis label only on the left panel
        if idx == 0:
            ax.set_ylabel("Gold label")
            ax.set_yticks(range(len(LABELS)))
            ax.set_yticklabels(display_labels)
        else:
            ax.set_ylabel("")
            ax.set_yticks(range(len(LABELS)))
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)

        ax.set_xticks(range(len(LABELS)))
        ax.set_xticklabels(display_labels)

        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                value = matrix[i, j]
                text_color = "white" if value >= 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.85)
    colorbar.set_label("Proportion")

    fig.savefig(
        out_dir / "best_individual_confusion_matrix_dev_test.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "best_individual_confusion_matrix_dev_test.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def export_label_behaviour_outputs(all_df: pd.DataFrame, wide_df: pd.DataFrame, paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_dir = paths["label_behavior_dir"]

    selected_df, recall_df, best_individual = build_per_class_recall_for_best_family_configs(all_df, wide_df)
    conf_df = build_confusion_rows_for_config(all_df, best_individual)

    selected_df.to_csv(out_dir / "label_level_selected_configurations.csv", index=False)
    recall_df.to_csv(out_dir / "per_class_recall_best_family_configs.csv", index=False)
    conf_df.to_csv(out_dir / "best_individual_confusion_matrices.csv", index=False)

    best_individual_df = pd.DataFrame([{**best_individual}])
    best_individual_df.to_csv(out_dir / "best_individual_selected_configuration.csv", index=False)

    with open(out_dir / "per_class_recall_best_family_configs.txt", "w", encoding="utf-8") as f:
        table_cols = [
            "split", "method_family_short", "selected_vlm", "selected_prompt",
            "selected_label_mode_display", "accuracy", "macro_f1",
            "entailment_recall", "neutral_recall", "contradiction_recall",
        ]
        f.write("Per-class recall for the best development-selected configuration of each method family\n")
        f.write("=" * 120 + "\n")
        f.write(recall_df[table_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        f.write("\n")

    plot_recall_heatmaps(recall_df, out_dir)
    plot_best_individual_confusion_matrices(conf_df, out_dir)

    return selected_df, recall_df, conf_df


# ============================================================
# 10. REPORT
# ============================================================

def write_report(
    paths: Dict[str, Path],
    all_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    method_summary_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    label_summary_df: pd.DataFrame,
    atomicity_df: pd.DataFrame,
    label_recall_df: pd.DataFrame,
):
    lines = []
    lines.append(SEP)
    lines.append("CANDIDATE PREDICTION ANALYSIS REPORT")
    lines.append(SEP)
    lines.append("")
    lines.append(f"Output root: {paths['output_root']}")
    lines.append("")

    lines.append(SEP)
    lines.append("INPUT SUMMARY")
    lines.append(SEP)
    lines.append(f"Total row-level prediction records loaded: {len(all_df):,}")
    lines.append("Records are loaded for dev/test, 2 VLMs, 2 prompts, and 4 method families.")
    lines.append("Independent atomic generated labels are aggregated using the strict NLI-style rule; independent atomic score-based labels are derived by summing atom-level label-likelihood scores and taking the maximum class.")
    lines.append("")

    lines.append(SEP)
    lines.append("INDIVIDUAL CANDIDATE PERFORMANCE")
    lines.append(SEP)
    cols = [
        "method_family",
        "selected_configuration",
        "dev_best_accuracy", "dev_best_macro_f1",
        "test_best_accuracy", "test_best_macro_f1",
        "dev_accuracy_range", "test_accuracy_range",
        "fusion_pool",
    ]
    lines.append(method_summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.append("")
    lines.append("Full 16-row configuration table saved in the Prediction Method Comparison folder.")
    lines.append("")

    lines.append(SEP)
    lines.append("CANDIDATE LABEL DERIVATION STABILITY")
    lines.append(SEP)
    lines.append("Full 32-point table saved in the Candidate Label Derivation Stability folder.")
    summary_cols = [
        "method_family",
        "generated_mean_accuracy", "generated_range_accuracy",
        "score_based_mean_accuracy", "score_based_range_accuracy",
        "better_mean_rule", "more_stable_rule",
    ]
    lines.append(label_summary_df[summary_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.append("")

    lines.append(SEP)
    lines.append("ATOM-COUNT ANALYSIS")
    lines.append(SEP)
    lines.append("Per-instance matched atom-bucket accuracy table saved in the Atomicity Analysis folder.")
    lines.append(atomicity_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.append("")

    lines.append(SEP)
    lines.append("LABEL-LEVEL BEHAVIOUR")
    lines.append(SEP)
    lines.append("Per-class recall heatmap and best-candidate confusion matrices saved in the Label-Level Behavior folder.")
    lines.append(label_recall_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lines.append("")

    lines.append(SEP)
    lines.append("OUTPUT FILES")
    lines.append(SEP)
    for key, path in paths.items():
        lines.append(f"{key:<30}: {path}")

    with open(paths["report_txt"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 11. MAIN
# ============================================================

def main():
    args = parse_args()
    paths = build_paths(args.base_dir)

    print(SEP)
    print("CANDIDATE PREDICTION ANALYSIS")
    print(SEP)
    print(f"Base dir   : {args.base_dir}")
    print(f"Output root: {paths['output_root']}")
    print("")

    # Load row-level predictions.
    all_df = load_all_predictions(paths["base_dir"])
    all_df.to_csv(paths["output_root"] / "all_row_level_predictions.csv", index=False)

    # Build metrics and summary tables.
    long_metrics_df = build_long_metric_table(all_df)
    long_metrics_df.to_csv(paths["output_root"] / "all_configuration_metrics_long.csv", index=False)

    wide_df = build_base_config_wide_table(long_metrics_df)
    method_summary_df = build_method_family_summary(wide_df)
    stability_df = build_label_derivation_stability_table(wide_df)
    label_summary_df = build_label_derivation_summary(stability_df)

    # Section-specific outputs.
    export_prediction_method_outputs(all_df, wide_df, method_summary_df, paths)
    export_label_derivation_outputs(stability_df, label_summary_df, paths)
    atomicity_df = export_atomicity_outputs(all_df, wide_df, paths)
    _, label_recall_df, _ = export_label_behaviour_outputs(all_df, wide_df, paths)

    # Report.
    write_report(
        paths=paths,
        all_df=all_df,
        wide_df=wide_df,
        method_summary_df=method_summary_df,
        stability_df=stability_df,
        label_summary_df=label_summary_df,
        atomicity_df=atomicity_df,
        label_recall_df=label_recall_df,
    )

    print("")
    print(SEP)
    print("CANDIDATE PREDICTION ANALYSIS COMPLETE")
    print(SEP)
    print(f"Output root: {paths['output_root']}")
    print(f"Report     : {paths['report_txt']}")
    print("")
    print("Main figures:")
    print(f"  {paths['label_derivation_dir'] / 'candidate_label_derivation_stability_scatter_updated.png'}")
    print(f"  {paths['atomicity_dir'] / 'atom_bucket_accuracy_dev_test_line.png'}")
    print(f"  {paths['label_behavior_dir'] / 'per_class_recall_heatmap_dev_test.png'}")
    print(f"  {paths['label_behavior_dir'] / 'best_individual_confusion_matrix_dev_test.png'}")


if __name__ == "__main__":
    main()
