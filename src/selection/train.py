"""Train the AVE-LS learned selector.

    python -m src.selection.train

Builds meta-features from the K=12 training prediction pool, compares four
feature variants against seven classifier configurations on one fixed
stratified split, and saves the selected model.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import config
from src.selection import features as F
from src.selection.classifiers import (
    HAS_XGBOOST,
    build_classifier_configs,
    classifier_family,
    export_hyperparameter_table,
)

SEP = "=" * 140
DASH = "-" * 140


def build_paths(output_name):
    train_dataset_dir = os.path.join(config.OUTPUT_DIR, "train_dataset")
    output_dir = os.path.join(config.OUTPUT_DIR, output_name)

    paths = {
        "train_dataset_dir": train_dataset_dir,
        "train_input_jsonl": os.path.join(train_dataset_dir,
                                          "decompose_atoms_qwen32_train.jsonl"),
        "qwen_train_dir": os.path.join(train_dataset_dir, "qwen3_predictions_v2"),
        "internvl_train_dir": os.path.join(train_dataset_dir, "internvl_predictions_v2"),
        "output_dir": output_dir,
        "results_dir": os.path.join(output_dir, "results"),
        "plots_dir": os.path.join(output_dir, "plots"),
        "reports_dir": os.path.join(output_dir, "reports"),
        "models_dir": os.path.join(output_dir, "saved_models"),
        "features_dir": os.path.join(output_dir, "feature_columns"),
        "predictions_dir": os.path.join(output_dir, "validation_predictions"),
    }

    for key in ["output_dir", "results_dir", "plots_dir", "reports_dir",
                "models_dir", "features_dir", "predictions_dir"]:
        os.makedirs(paths[key], exist_ok=True)

    r = paths["results_dir"]
    paths.update({
        "all_results_csv": os.path.join(r, "all_tuning_results.csv"),
        "best_per_variant_csv": os.path.join(r, "best_per_feature_variant.csv"),
        "compact_summary_csv": os.path.join(r, "compact_summary_table.csv"),
        "hyperparameter_configs_csv": os.path.join(r, "hyperparameter_configs.csv"),
        "hyperparameter_configs_tex": os.path.join(r, "hyperparameter_configs_latex.tex"),
        "selected_config_json": os.path.join(
            r, "selected_ave_ls_label_selector_v3_config.json"),
        "selected_val_jsonl": os.path.join(paths["predictions_dir"],
                                           "selected_validation_predictions.jsonl"),
        "report_txt": os.path.join(paths["reports_dir"], "experiment_summary.txt"),
    })
    return paths


def get_folder_for_method(method_key, paths):
    if method_key.startswith("qwen"):
        return paths["qwen_train_dir"]
    if method_key.startswith("internvl"):
        return paths["internvl_train_dir"]
    raise ValueError(f"Unknown method key: {method_key}")


def load_all_required_methods(paths):
    loaded = {}
    print(SEP)
    print("Loading TRAIN prediction files")
    print(SEP)

    for method_key in F.ALL_METHOD_KEYS:
        folder = get_folder_for_method(method_key, paths)
        path = F.resolve_file(folder, F.FILE_CANDIDATES[method_key])
        data = F.load_prediction_map(path, F.method_kind(method_key))
        loaded[method_key] = data
        print(f"{method_key:<36}: {len(data):>7} rows | {path}")

    print("")
    return loaded


def compute_global_common_keys(metadata, loaded_methods):
    common = set(metadata.keys())
    for method_key in F.ALL_METHOD_KEYS:
        common &= set(loaded_methods[method_key].keys())
    return sorted(common, key=lambda k: metadata[k]["order"])


def build_dataset_for_variant(variant_name, global_keys, metadata, loaded_methods):
    method_keys = F.FEATURE_VARIANTS[variant_name]
    X_rows, y_rows, meta_rows, outputs_rows = [], [], [], []

    for key in global_keys:
        method_outputs = {mk: loaded_methods[mk][key] for mk in method_keys}
        X_rows.append(F.build_feature_row(metadata[key], method_outputs, method_keys))
        y_rows.append(metadata[key]["gold"])
        meta_rows.append({**metadata[key], "key": key})
        outputs_rows.append(method_outputs)

    X = pd.DataFrame(X_rows).fillna(0.0)
    y = pd.Series(y_rows)
    # Stable column order for reproducibility.
    X = X.reindex(sorted(X.columns), axis=1)

    print(f"{variant_name}:")
    print(f"  common examples: {len(X)}")
    print(f"  feature count  : {X.shape[1] if len(X) else 0}")
    print(f"  gold dist      : {dict(Counter(y))}")
    print("")

    return X, y, meta_rows, outputs_rows


def metric_report(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=F.LABELS, average="macro", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=F.LABELS, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=F.LABELS)

    return {
        "n": len(y_true),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(s[i])}
                      for i, lab in enumerate(F.LABELS)},
        "confusion_matrix": cm.tolist(),
        "prediction_distribution": dict(Counter(y_pred)),
    }


def make_validation_predictions(variant_name, classifier_config, model, X_val,
                                y_val_ids, val_indices, meta_rows,
                                outputs_rows, method_keys):
    pred_ids = model.predict(X_val)
    learned_labels = [F.ID_TO_LABEL[int(i)] for i in pred_ids]
    true_labels = [F.ID_TO_LABEL[int(i)] for i in y_val_ids]
    rows = []

    for local_idx, original_idx in enumerate(val_indices):
        original_idx = int(original_idx)
        learned_label = F.normalize_label(learned_labels[local_idx])
        gold = F.normalize_label(true_labels[local_idx])
        outputs = outputs_rows[original_idx]

        selected_method, alignment_reason = F.align_candidate_to_label(
            learned_label, outputs, method_keys)
        selected_output = outputs[selected_method]
        selected_candidate_label = F.normalize_label(
            selected_output.get("prediction", "neutral"))
        selected_scores = F.scores_to_probs(selected_output.get("scores", {}))
        selected_diag = F.confidence_diag(selected_scores)

        rows.append({
            "feature_variant": variant_name,
            "classifier_config": classifier_config,
            "classifier_family": classifier_family(classifier_config),
            "Flickr30K_ID": meta_rows[original_idx]["Flickr30K_ID"],
            "hypothesis": meta_rows[original_idx]["hypothesis"],
            "gold": gold,
            "learned_label": learned_label,
            "final_label": learned_label,
            "selected_candidate": selected_method,
            "selected_candidate_label": selected_candidate_label,
            "selected_candidate_score_prediction": F.normalize_label(
                selected_output.get("score_prediction", selected_candidate_label)),
            "selected_candidate_confidence": float(selected_output.get(
                "confidence_score", selected_diag["confidence_score"])),
            "selected_candidate_margin": float(selected_output.get(
                "margin", selected_diag["margin"])),
            "selected_candidate_scores": selected_scores,
            "alignment_reason": alignment_reason,
            "candidate_matches_learned_label": int(
                selected_candidate_label == learned_label),
            "final_correct": int(learned_label == gold),
        })

    return true_labels, learned_labels, rows


def train_and_validate_model(variant_name, classifier_config, classifier_template,
                             X, y, train_idx, val_idx, meta_rows, outputs_rows, paths):
    y_ids = y.map(F.LABEL_TO_ID).values

    start = time.time()
    model = clone(classifier_template)
    model.fit(X.iloc[train_idx], y_ids[train_idx])
    elapsed = time.time() - start

    y_true, y_pred, pred_rows = make_validation_predictions(
        variant_name, classifier_config, model, X.iloc[val_idx],
        y_ids[val_idx], val_idx, meta_rows, outputs_rows,
        F.FEATURE_VARIANTS[variant_name])

    res = metric_report(y_true, y_pred)

    pred_path = os.path.join(
        paths["predictions_dir"],
        f"{variant_name}__{classifier_config}__validation_predictions.jsonl")
    F.write_jsonl(pred_path, pred_rows)

    candidate_match_rate = (
        float(np.mean([r["candidate_matches_learned_label"] for r in pred_rows]))
        if pred_rows else 0.0)

    return {
        "feature_variant": variant_name,
        "feature_variant_display": F.VARIANT_DISPLAY_NAMES.get(variant_name, variant_name),
        "classifier_config": classifier_config,
        "classifier_family": classifier_family(classifier_config),
        "num_common_examples": len(X),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "feature_count": int(X.shape[1]),
        "fit_seconds": float(elapsed),
        "val_accuracy": res["accuracy"],
        "val_balanced_accuracy": res["balanced_accuracy"],
        "val_macro_precision": res["macro_precision"],
        "val_macro_recall": res["macro_recall"],
        "val_macro_f1": res["macro_f1"],
        "val_entailment_recall": res["per_class"]["entailment"]["recall"],
        "val_neutral_recall": res["per_class"]["neutral"]["recall"],
        "val_contradiction_recall": res["per_class"]["contradiction"]["recall"],
        "candidate_match_rate": candidate_match_rate,
        "prediction_distribution": json.dumps(res["prediction_distribution"]),
        "confusion_matrix": json.dumps({"labels": F.LABELS,
                                        "matrix": res["confusion_matrix"]}),
        "validation_predictions_path": pred_path,
    }


def retrain_and_save_model(variant_name, classifier_config, classifier_template,
                           X, y, paths):
    y_ids = y.map(F.LABEL_TO_ID).values
    model = clone(classifier_template)
    model.fit(X, y_ids)

    model_path = os.path.join(paths["models_dir"],
                              f"{variant_name}__{classifier_config}.joblib")
    columns_path = os.path.join(paths["features_dir"],
                                f"{variant_name}__feature_columns.json")

    joblib.dump(model, model_path)
    with open(columns_path, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    return {"model_path": model_path, "feature_columns_path": columns_path}


def make_compact_tables(results_df, paths):
    results_df.to_csv(paths["all_results_csv"], index=False)

    best_rows = []
    for variant_name in F.VARIANT_DISPLAY_ORDER:
        sub = results_df[results_df["feature_variant"] == variant_name]
        if sub.empty:
            continue
        best = sub.sort_values(["val_macro_f1", "val_accuracy"],
                               ascending=[False, False]).iloc[0]
        best_rows.append(best.to_dict())

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(paths["best_per_variant_csv"], index=False)

    compact_cols = ["feature_variant_display", "classifier_config", "classifier_family",
                    "feature_count", "val_accuracy", "val_macro_f1"]
    results_df[compact_cols].to_csv(paths["compact_summary_csv"], index=False)

    return best_df, results_df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-name", default="AVE_train_learned_selector_v3")
    ap.add_argument("--validation-size", type=float, default=0.20)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    paths = build_paths(args.output_name)
    classifier_configs = build_classifier_configs(args.random_state)

    print(SEP)
    print("TRAIN AVE-LS LABEL SELECTOR V3")
    print(SEP)
    print(f"Train metadata : {paths['train_input_jsonl']}")
    print(f"Qwen train dir : {paths['qwen_train_dir']}")
    print(f"InternVL dir   : {paths['internvl_train_dir']}")
    print(f"Output dir     : {paths['output_dir']}")
    print(f"XGBoost available: {HAS_XGBOOST}")
    print("")

    export_hyperparameter_table(classifier_configs,
                                paths["hyperparameter_configs_csv"],
                                paths["hyperparameter_configs_tex"])

    train_metadata = F.load_dataset_metadata(paths["train_input_jsonl"])
    print(f"Loaded train metadata: {len(train_metadata)}\n")

    loaded_methods = load_all_required_methods(paths)
    global_keys = compute_global_common_keys(train_metadata, loaded_methods)

    print(SEP)
    print("Global common key set")
    print(SEP)
    print(f"Global common examples across metadata + all 12 method files: "
          f"{len(global_keys)}\n")

    if not global_keys:
        raise RuntimeError("No global common keys found. Check file keys and paths.")

    print(SEP)
    print("Building feature variants on the same global common examples")
    print(SEP)

    variant_data = {}
    for variant_name in F.VARIANT_DISPLAY_ORDER:
        X, y, meta, outputs = build_dataset_for_variant(
            variant_name, global_keys, train_metadata, loaded_methods)
        variant_data[variant_name] = {"X": X, "y": y, "meta": meta, "outputs": outputs}

    # One fixed stratified split shared by all variants and configs.
    y_global = variant_data[F.VARIANT_DISPLAY_ORDER[0]]["y"]
    y_global_ids = y_global.map(F.LABEL_TO_ID).values
    train_idx, val_idx = train_test_split(
        np.arange(len(global_keys)),
        test_size=args.validation_size,
        stratify=y_global_ids,
        random_state=args.random_state,
    )
    train_idx = np.array(sorted(train_idx), dtype=int)
    val_idx = np.array(sorted(val_idx), dtype=int)

    print(SEP)
    print("Fixed stratified split")
    print(SEP)
    print(f"Train rows     : {len(train_idx)}")
    print(f"Validation rows: {len(val_idx)}")
    print(f"Train dist     : {dict(Counter(y_global.iloc[train_idx]))}")
    print(f"Val dist       : {dict(Counter(y_global.iloc[val_idx]))}\n")

    print(SEP)
    print("Training and validating classifier configs")
    print(SEP)

    result_rows = []
    for variant_name in F.VARIANT_DISPLAY_ORDER:
        data = variant_data[variant_name]
        for config_name, template in classifier_configs.items():
            print(f"Training validation model: {variant_name} / {config_name}")
            row = train_and_validate_model(
                variant_name, config_name, template, data["X"], data["y"],
                train_idx, val_idx, data["meta"], data["outputs"], paths)
            result_rows.append(row)
            print(f"  val_acc={row['val_accuracy']:.4f} "
                  f"val_macro_f1={row['val_macro_f1']:.4f} "
                  f"time={row['fit_seconds']:.1f}s")

    results_df = pd.DataFrame(result_rows).sort_values(
        ["val_macro_f1", "val_accuracy"], ascending=[False, False]
    ).reset_index(drop=True)
    results_df["selected_overall"] = False
    results_df.loc[0, "selected_overall"] = True

    make_compact_tables(results_df, paths)

    selected_row = results_df.iloc[0].to_dict()
    selected_variant = selected_row["feature_variant"]
    selected_classifier = selected_row["classifier_config"]

    print("")
    print(SEP)
    print("Selected AVE-LS label selector V3")
    print(SEP)
    print(f"Feature variant : {selected_variant}")
    print(f"Classifier      : {selected_classifier}")
    print(f"Classifier family: {classifier_family(selected_classifier)}")
    print(f"Val accuracy    : {selected_row['val_accuracy']:.4f}")
    print(f"Val macro F1    : {selected_row['val_macro_f1']:.4f}\n")

    saved = {}
    for variant_name in F.VARIANT_DISPLAY_ORDER:
        sub = results_df[results_df["feature_variant"] == variant_name]
        if sub.empty:
            continue
        best = sub.sort_values(["val_macro_f1", "val_accuracy"],
                               ascending=[False, False]).iloc[0]
        data = variant_data[variant_name]
        info = retrain_and_save_model(
            variant_name, best["classifier_config"],
            classifier_configs[best["classifier_config"]],
            data["X"], data["y"], paths)
        saved[variant_name] = info
        print(f"Saved {variant_name}: {info['model_path']}")

    selected_config = {
        "selected_feature_variant": selected_variant,
        "selected_classifier_config": selected_classifier,
        "selected_classifier_family": classifier_family(selected_classifier),
        "validation_accuracy": selected_row["val_accuracy"],
        "validation_macro_f1": selected_row["val_macro_f1"],
        "feature_count": selected_row["feature_count"],
        "num_common_examples": selected_row["num_common_examples"],
        "train_rows": selected_row["train_rows"],
        "validation_rows": selected_row["validation_rows"],
        "validation_size": args.validation_size,
        "random_state": args.random_state,
        "selection_metric_primary": "validation_macro_f1",
        "selection_metric_secondary": "validation_accuracy",
        "method_keys": F.FEATURE_VARIANTS[selected_variant],
        "model_path": saved[selected_variant]["model_path"],
        "feature_columns_path": saved[selected_variant]["feature_columns_path"],
    }
    with open(paths["selected_config_json"], "w", encoding="utf-8") as f:
        json.dump(selected_config, f, indent=2)

    print(f"\nWritten to {paths['output_dir']}")


if __name__ == "__main__":
    main()