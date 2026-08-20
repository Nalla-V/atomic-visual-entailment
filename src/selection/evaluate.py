"""Evaluate the trained AVE-LS selector on dev or test.

    python -m src.selection.evaluate --split dev

Loads the saved model and its feature columns, rebuilds meta-features from the
cleaned prediction files, reports accuracy and macro-F1, and writes the input
file used by the grounding stage.
"""

import argparse
import json
import os
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import config
from src.selection import features as F
from src.selection.train import metric_report

SEP = "=" * 140
DASH = "-" * 140

GROUNDABLE_FINAL_LABELS = {"entailment", "contradiction"}


# ============================================================
# PATHS
# ============================================================

def build_paths(split, train_run_name, output_name,
                qwen_dir_name="qwen3_predictions_clean_v2",
                internvl_dir_name="internvl_predictions_clean_v2"):
    if split not in {"dev", "test"}:
        raise ValueError("split must be either 'dev' or 'test'")

    dataset_dir = os.path.join(config.OUTPUT_DIR, f"{split}_dataset")
    output_dir = os.path.join(dataset_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(config.OUTPUT_DIR, train_run_name)

    return {
        "split": split,
        "dataset_dir": dataset_dir,
        "input_jsonl": os.path.join(dataset_dir, f"decompose_atoms_qwen32_{split}.jsonl"),
        "qwen_dir": os.path.join(dataset_dir, qwen_dir_name),
        "internvl_dir": os.path.join(dataset_dir, internvl_dir_name),
        "model_dir": model_dir,
        "selected_config_json": os.path.join(
            model_dir, "results", "selected_ave_ls_label_selector_v3_config.json"),
        "output_dir": output_dir,
        "summary_csv": os.path.join(output_dir, f"ave_ls_v3_{split}_summary.csv"),
        "confusion_csv": os.path.join(output_dir, f"ave_ls_v3_{split}_confusion_matrices.csv"),
        "predictions_jsonl": os.path.join(output_dir, f"ave_ls_v3_{split}_predictions.jsonl"),
        "grounding_jsonl": os.path.join(output_dir, f"ave_ls_v3_{split}_grounding_input.jsonl"),
        "feature_column_check_json": os.path.join(
            output_dir, f"ave_ls_v3_{split}_feature_column_check.json"),
        "report_txt": os.path.join(output_dir, f"ave_ls_v3_{split}_report.txt"),
    }


def build_file_candidates(split):
    return {
        "qwen_baseline_simple": [f"baseline_simple_{split}.jsonl"],
        "qwen_baseline_structured": [f"baseline_structured_{split}.jsonl"],
        "qwen_joint_simple": [f"atomic_joint_simple_{split}.jsonl"],
        "qwen_joint_structured": [f"atomic_joint_structured_{split}.jsonl"],
        "qwen_self_decompose_simple": [f"self_decompose_simple_{split}.jsonl"],
        "qwen_self_decompose_structured": [f"self_decompose_structured_{split}.jsonl"],
        "internvl_baseline_simple": [f"baseline_simple_{split}.jsonl"],
        "internvl_baseline_structured": [f"baseline_structured_{split}.jsonl"],
        "internvl_joint_simple": [f"atomic_joint_simple_{split}.jsonl"],
        "internvl_joint_structured": [f"atomic_joint_structured_{split}.jsonl"],
        "internvl_self_decompose_simple": [f"self_decompose_simple_{split}.jsonl"],
        "internvl_self_decompose_structured": [f"self_decompose_structured_{split}.jsonl"],
    }


# ============================================================
# LOADING
# ============================================================

def make_default_output(method_key, key):
    """A candidate row missing for this key becomes a neutral, uniform vote."""
    scores = {lab: 1.0 / len(F.LABELS) for lab in F.LABELS}
    diag = F.confidence_diag(scores)
    return {
        "prediction": "neutral",
        "score_prediction": "neutral",
        "scores": scores,
        "reason": "",
        "parse_ok": False,
        "parse_error": "missing prediction row for this key",
        "missing_candidate_output": True,
        "raw_result": {},
        "source_row_key": key,
        **diag,
    }


def load_reference_rows(path):
    rows = F.read_jsonl(path)
    out = []
    seen = Counter()

    for i, row in enumerate(rows):
        img_id = F.safe_text(row.get("Flickr30K_ID", ""))
        hypo = F.safe_text(row.get("hypothesis", row.get("sentence2", "")))
        gold = F.normalize_label(row.get("annotator_label", row.get("gold", "")))

        atoms = F.ensure_atom_texts(row.get("atomic_facts", row.get("raw_atoms", [])))
        if not atoms:
            atoms = [hypo]

        key = F.make_key_from_values(img_id, hypo, gold)
        seen[key] += 1

        out.append({
            "row_id": i,
            "row_key_occurrence": seen[key],
            "key": key,
            "Flickr30K_ID": img_id,
            "hypothesis": hypo,
            "annotator_label": gold,
            "gold": gold,
            "atomic_facts": atoms,
            "num_atoms": len(atoms),
            "atom_bucket": F.atom_bucket(len(atoms)),
            "hypothesis_word_count": F.word_count(hypo),
            "total_atom_word_count": sum(F.word_count(a) for a in atoms),
        })

    return out


def get_folder_for_method(method_key, qwen_dir, internvl_dir):
    if method_key.startswith("qwen"):
        return qwen_dir
    if method_key.startswith("internvl"):
        return internvl_dir
    raise ValueError(f"Unknown method key: {method_key}")


def load_all_required_methods(method_keys, split, paths):
    file_candidates = build_file_candidates(split)
    loaded = {}

    print(SEP)
    print(f"Loading {split.upper()} prediction files")
    print(SEP)

    for method_key in method_keys:
        folder = get_folder_for_method(method_key, paths["qwen_dir"], paths["internvl_dir"])
        path = F.resolve_file(folder, file_candidates[method_key])
        data = F.load_prediction_map(path, F.method_kind(method_key))
        loaded[method_key] = data
        print(f"{method_key:<36}: {len(data):>7} rows | {path}")

    print("")
    return loaded


def build_dataset_from_reference_rows(reference_rows, method_keys, loaded_methods):
    X_rows, y_rows, meta_rows, outputs_rows = [], [], [], []
    missing_counts = Counter()

    for meta in reference_rows:
        key = meta["key"]
        method_outputs = {}
        for method_key in method_keys:
            out = loaded_methods.get(method_key, {}).get(key)
            if out is None:
                out = make_default_output(method_key, key)
                missing_counts[method_key] += 1
            method_outputs[method_key] = out

        X_rows.append(F.build_feature_row(meta, method_outputs, method_keys))
        y_rows.append(meta["gold"])
        meta_rows.append(meta)
        outputs_rows.append(method_outputs)

    if missing_counts:
        print("WARNING: missing candidate outputs were replaced with neutral defaults:")
        for method_key, count in missing_counts.items():
            print(f"  {method_key:<36}: {count}")
        print("")

    return (pd.DataFrame(X_rows).fillna(0.0), pd.Series(y_rows),
            meta_rows, outputs_rows)


def validate_feature_columns(X, feature_columns, paths):
    generated = list(X.columns)
    missing_cols = sorted(set(feature_columns) - set(generated))
    extra_cols = sorted(set(generated) - set(feature_columns))

    check = {
        "generated_feature_count": len(generated),
        "required_feature_count": len(feature_columns),
        "missing_columns": missing_cols,
        "extra_columns": extra_cols,
    }
    with open(paths["feature_column_check_json"], "w", encoding="utf-8") as f:
        json.dump(check, f, indent=2)

    if missing_cols:
        print(f"WARNING: {len(missing_cols)} required columns are missing and "
              f"will be filled with 0.0")
    if extra_cols:
        print(f"Note: {len(extra_cols)} extra generated columns will be ignored "
              f"after reindexing.")
    return check


# ============================================================
# GROUNDING INPUT
# ============================================================

def parse_method_key(method_key):
    if method_key.startswith("qwen"):
        model = "qwen3"
    elif method_key.startswith("internvl"):
        model = "internvl3"
    else:
        model = "unknown"

    if "baseline" in method_key:
        method, result_key = "baseline", "full_hypothesis_results"
    elif "joint" in method_key:
        method, result_key = "joint_atomic", "joint_atom_results"
    elif "self_decompose" in method_key:
        method, result_key = "self_decompose", "self_decompose_results"
    else:
        method, result_key = "unknown", ""

    if "structured" in method_key:
        prompt = "structured"
    elif "simple" in method_key:
        prompt = "simple"
    else:
        prompt = "unknown"

    return {
        "candidate_key": method_key,
        "selected_candidate": method_key,
        "selected_model": model,
        "selected_method": method,
        "selected_prompt": prompt,
        "result_key": result_key,
    }


def normalize_atom_observation(obs, default_label="neutral"):
    return {
        "atom": F.safe_text(obs.get("atom", obs.get("claim",
                            obs.get("fact", obs.get("text", ""))))),
        "atom_label": F.normalize_label(obs.get("label",
                                        obs.get("status", default_label))),
        "vlm_reasoning": F.safe_text(obs.get("reason", obs.get("evidence",
                                     obs.get("visible_evidence", "")))),
    }


def extract_self_decompose_atoms(raw_result):
    atoms = raw_result.get("decomposed_atoms", [])
    if not isinstance(atoms, list):
        return []

    cleaned = []
    for idx, item in enumerate(atoms):
        if isinstance(item, dict):
            obs = normalize_atom_observation(item)
            if obs["atom"]:
                cleaned.append({"atom_index": item.get("atom_index", idx), **obs})
        elif isinstance(item, str):
            text = F.safe_text(item)
            if text:
                cleaned.append({"atom_index": idx, "atom": text,
                                "atom_label": "neutral", "vlm_reasoning": ""})
    return cleaned


def fallback_evidence_item(meta, final_label, full_reason, evidence_source):
    return [{
        "evidence_index": 1,
        "atom": F.safe_text(meta.get("hypothesis", "")),
        "atom_label": final_label,
        "vlm_reasoning": full_reason,
        "evidence_source": evidence_source,
    }]


def build_evidence_items_for_selected_output(selected_candidate, selected_output,
                                             meta, final_label):
    """Method-agnostic evidence items for the phrase extractor.

    Joint atomic uses matching atom_observations, self-decompose uses matching
    decomposed_atoms, and baseline treats the hypothesis as a single evidence
    unit carrying the sentence-level reason.
    """
    final_label = F.normalize_label(final_label)
    selected_method = parse_method_key(selected_candidate)["selected_method"]

    raw_result = selected_output.get("raw_result", {}) or {}
    if not isinstance(raw_result, dict):
        raw_result = {}

    full_reason = F.safe_text(selected_output.get("reason",
                                                  raw_result.get("reason", "")))
    evidence_items = []

    if selected_method == "joint_atomic":
        observations = raw_result.get("atom_observations", [])
        if isinstance(observations, list):
            for obs in observations:
                if not isinstance(obs, dict):
                    continue
                clean = normalize_atom_observation(obs)
                if not clean["atom"]:
                    continue
                if clean["atom_label"] == final_label:
                    evidence_items.append({
                        "evidence_index": len(evidence_items) + 1,
                        **clean,
                        "evidence_source": "selected_joint_atom_observation",
                    })
        if evidence_items:
            return evidence_items[:4], full_reason
        return fallback_evidence_item(
            meta, final_label, full_reason,
            "selected_joint_final_reason_fallback"), full_reason

    if selected_method == "self_decompose":
        self_atoms = extract_self_decompose_atoms(raw_result)
        for item in self_atoms:
            if item["atom_label"] == final_label:
                evidence_items.append({
                    "evidence_index": len(evidence_items) + 1,
                    "atom": item["atom"],
                    "atom_label": item["atom_label"],
                    "vlm_reasoning": item.get("vlm_reasoning", ""),
                    "evidence_source": "selected_self_decompose_atom",
                })
        if evidence_items:
            return evidence_items[:4], full_reason
        if self_atoms:
            # No atom carries the final label, so keep the first one and
            # override its label with the final label.
            first = self_atoms[0]
            return [{
                "evidence_index": 1,
                "atom": first["atom"],
                "atom_label": final_label,
                "vlm_reasoning": full_reason or first.get("vlm_reasoning", ""),
                "evidence_source": "selected_self_decompose_final_reason_fallback",
            }], full_reason
        return fallback_evidence_item(
            meta, final_label, full_reason,
            "selected_self_decompose_hypothesis_fallback"), full_reason

    # Baseline has no atom observations, so the hypothesis is the evidence unit.
    return [{
        "evidence_index": 1,
        "atom": F.safe_text(meta.get("hypothesis", "")),
        "atom_label": final_label,
        "vlm_reasoning": full_reason,
        "evidence_source": "selected_baseline_full_hypothesis_reason",
    }], full_reason


def grounding_filter_reason(final_label, candidate_matches_learned_label):
    final_label = F.normalize_label(final_label)
    if not candidate_matches_learned_label:
        return False, "selected_candidate_label_does_not_match_learned_final_label"
    if final_label not in GROUNDABLE_FINAL_LABELS:
        return False, "final_label_is_neutral_not_grounded"
    return True, "candidate_matched_and_final_label_is_entailment_or_contradiction"


# ============================================================
# OUTPUT
# ============================================================

def save_confusion_csv(path, matrix):
    rows = []
    for i, gold in enumerate(F.LABELS):
        row = {"gold": gold}
        for j, pred in enumerate(F.LABELS):
            row[f"pred_{pred}"] = int(matrix[i][j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================
# MAIN
# ============================================================

def run(split, train_run_name, output_name):
    paths = build_paths(split, train_run_name, output_name)

    print(SEP)
    print(f"EVALUATE AVE-LS ON {split.upper()}")
    print(SEP)
    print(f"Reference  : {paths['input_jsonl']}")
    print(f"Qwen dir   : {paths['qwen_dir']}")
    print(f"InternVL   : {paths['internvl_dir']}")
    print(f"Model dir  : {paths['model_dir']}")
    print(f"Output dir : {paths['output_dir']}\n")

    with open(paths["selected_config_json"], "r", encoding="utf-8") as f:
        selected_config = json.load(f)

    method_keys = selected_config["method_keys"]
    model_path = selected_config["model_path"]
    columns_path = selected_config["feature_columns_path"]

    print(f"Feature variant : {selected_config['selected_feature_variant']}")
    print(f"Classifier      : {selected_config['selected_classifier_config']}")
    print(f"Model           : {model_path}\n")

    model = joblib.load(model_path)
    with open(columns_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    reference_rows = load_reference_rows(paths["input_jsonl"])
    print(f"Reference rows: {len(reference_rows)}\n")

    loaded_methods = load_all_required_methods(method_keys, split, paths)

    X, y, meta_rows, outputs_rows = build_dataset_from_reference_rows(
        reference_rows, method_keys, loaded_methods)

    validate_feature_columns(X, feature_columns, paths)
    X = X.reindex(columns=feature_columns, fill_value=0.0)

    pred_ids = model.predict(X)
    learned_labels = [F.ID_TO_LABEL[int(i)] for i in pred_ids]
    gold_labels = list(y)

    res = metric_report(gold_labels, learned_labels)

    print(SEP)
    print(f"AVE-LS results on {split}")
    print(SEP)
    print(f"n            : {res['n']}")
    print(f"Accuracy     : {res['accuracy']:.4f}")
    print(f"Macro-F1     : {res['macro_f1']:.4f}")
    print(f"Balanced acc : {res['balanced_accuracy']:.4f}")
    print(DASH)
    for lab in F.LABELS:
        pc = res["per_class"][lab]
        print(f"{lab:<15} precision={pc['precision']:.4f} "
              f"recall={pc['recall']:.4f} f1={pc['f1']:.4f} support={pc['support']}")
    print(DASH)

    # Per-row output, plus the subset that is eligible for grounding.
    pred_rows, grounding_rows = [], []
    grounding_eligible_count = 0
    filter_counter = Counter()

    for idx, meta in enumerate(meta_rows):
        learned_label = learned_labels[idx]
        outputs = outputs_rows[idx]

        selected_method, alignment_reason = F.align_candidate_to_label(
            learned_label, outputs, method_keys)
        selected_output = outputs[selected_method]
        selected_candidate_label = F.normalize_label(
            selected_output.get("prediction", "neutral"))
        selected_scores = F.scores_to_probs(selected_output.get("scores", {}))

        candidate_matches_learned = selected_candidate_label == learned_label
        eligible, filter_reason = grounding_filter_reason(
            learned_label, candidate_matches_learned)
        filter_counter[filter_reason] += 1

        identity = parse_method_key(selected_method)
        evidence_items, full_reason = build_evidence_items_for_selected_output(
            selected_method, selected_output, meta, learned_label)

        pred_rows.append({
            "row_id": meta["row_id"],
            "Flickr30K_ID": meta["Flickr30K_ID"],
            "hypothesis": meta["hypothesis"],
            "atomic_facts": meta["atomic_facts"],
            "gold": meta["gold"],
            "final_label": learned_label,
            "final_correct": int(learned_label == meta["gold"]),
            "selected_candidate": selected_method,
            "selected_candidate_label": selected_candidate_label,
            "selected_candidate_reason": full_reason,
            "selected_candidate_scores": selected_scores,
            "candidate_matches_learned_label": int(candidate_matches_learned),
            "grounding_eligible": bool(eligible),
            "grounding_filter_reason": filter_reason,
            "alignment_reason": alignment_reason,
        })

        if eligible:
            grounding_eligible_count += 1
            grounding_rows.append({
                "row_id": meta["row_id"],
                "row_key_occurrence": meta["row_key_occurrence"],
                "Flickr30K_ID": meta["Flickr30K_ID"],
                "hypothesis": meta["hypothesis"],
                "annotator_label": meta["gold"],
                "gold": meta["gold"],
                "final_label": learned_label,
                "prediction": learned_label,
                "selected_candidate": selected_method,
                "selected_model": identity["selected_model"],
                "selected_method": identity["selected_method"],
                "selected_prompt": identity["selected_prompt"],
                "selected_candidate_label": selected_candidate_label,
                "candidate_matches_learned_label": True,
                "grounding_eligible": True,
                "grounding_filter_reason": filter_reason,
                "atomic_facts": meta.get("atomic_facts", []),
                "reason": full_reason,
                "evidence_items": evidence_items,
                "selected_output": selected_output,
            })

    F.write_jsonl(paths["predictions_jsonl"], pred_rows)
    F.write_jsonl(paths["grounding_jsonl"], grounding_rows)

    print(f"\nGrounding eligible: {grounding_eligible_count} / {len(meta_rows)} "
          f"({grounding_eligible_count / len(meta_rows):.3f})")
    for reason, count in filter_counter.most_common():
        print(f"  {reason:<62} {count:>7}")
    print(f"\nGrounding input: {paths['grounding_jsonl']}")

    pd.DataFrame([{
        "split": split,
        "n": res["n"],
        "accuracy": res["accuracy"],
        "macro_f1": res["macro_f1"],
        "balanced_accuracy": res["balanced_accuracy"],
        "macro_precision": res["macro_precision"],
        "macro_recall": res["macro_recall"],
        "entailment_recall": res["per_class"]["entailment"]["recall"],
        "neutral_recall": res["per_class"]["neutral"]["recall"],
        "contradiction_recall": res["per_class"]["contradiction"]["recall"],
        "grounding_eligible_count": grounding_eligible_count,
        "grounding_eligible_rate": grounding_eligible_count / len(meta_rows),
        "feature_variant": selected_config["selected_feature_variant"],
        "classifier_config": selected_config["selected_classifier_config"],
    }]).to_csv(paths["summary_csv"], index=False)

    save_confusion_csv(paths["confusion_csv"], res["confusion_matrix"])

    with open(paths["report_txt"], "w", encoding="utf-8") as f:
        f.write(json.dumps({"split": split,
                            "grounding_eligible_count": grounding_eligible_count,
                            **{k: v for k, v in res.items() if k != "per_class"}},
                           indent=2))

    print(f"\nWritten to {paths['output_dir']}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--train-run-name", default="AVE_train_learned_selector_v3")
    ap.add_argument("--output-name", default="AVE_learned_selection_evaluation_v3")
    args = ap.parse_args()
    run(args.split, args.train_run_name, args.output_name)


if __name__ == "__main__":
    main()