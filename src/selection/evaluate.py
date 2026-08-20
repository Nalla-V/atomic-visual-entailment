"""Evaluate the trained AVE-LS selector on dev or test.

    python -m src.selection.evaluate --split dev

Loads the saved model and its feature columns, rebuilds meta-features from the
cleaned prediction files, and reports accuracy and macro-F1.
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


def save_confusion_csv(path, matrix):
    rows = []
    for i, gold in enumerate(F.LABELS):
        row = {"gold": gold}
        for j, pred in enumerate(F.LABELS):
            row[f"pred_{pred}"] = int(matrix[i][j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


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
        "feature_variant": selected_config["selected_feature_variant"],
        "classifier_config": selected_config["selected_classifier_config"],
    }]).to_csv(paths["summary_csv"], index=False)

    save_confusion_csv(paths["confusion_csv"], res["confusion_matrix"])

    pred_rows = []
    for idx, meta in enumerate(meta_rows):
        learned_label = learned_labels[idx]
        outputs = outputs_rows[idx]
        selected_method, alignment_reason = F.align_candidate_to_label(
            learned_label, outputs, method_keys)
        selected_output = outputs[selected_method]
        selected_scores = F.scores_to_probs(selected_output.get("scores", {}))

        pred_rows.append({
            "row_id": meta["row_id"],
            "Flickr30K_ID": meta["Flickr30K_ID"],
            "hypothesis": meta["hypothesis"],
            "atomic_facts": meta["atomic_facts"],
            "gold": meta["gold"],
            "final_label": learned_label,
            "final_correct": int(learned_label == meta["gold"]),
            "selected_candidate": selected_method,
            "selected_candidate_label": F.normalize_label(
                selected_output.get("prediction", "neutral")),
            "selected_candidate_reason": F.safe_text(selected_output.get("reason", "")),
            "selected_candidate_scores": selected_scores,
            "alignment_reason": alignment_reason,
        })

    F.write_jsonl(paths["predictions_jsonl"], pred_rows)

    with open(paths["report_txt"], "w", encoding="utf-8") as f:
        f.write(json.dumps({"split": split, **{k: v for k, v in res.items()
                                               if k != "per_class"}}, indent=2))

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