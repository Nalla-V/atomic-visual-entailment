"""Majority voting over the K=12 candidate prediction pool.

    python -m src.selection.voting

Reproduces the majority voting table: best individual candidates, AVE-MV intra
per VLM (K=6), AVE-MV inter over the full pool (K=12), and the oracle bound.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import config
from src.selection import candidates as cand
from src.selection.common import (
    LABELS,
    LABEL_TO_ID,
    compute_metrics,
    confusion_matrix_counts,
    ensure_dir,
    fmt_num,
    row_normalize,
    text_table,
    write_csv,
    write_jsonl,
)

SEP = "=" * 100
DASH = "-" * 100


def candidate_vector(data, candidate_id, decision="generated"):
    if decision not in {"generated", "score"}:
        raise ValueError(f"Unknown decision source: {decision}")
    label_key = "generated_label" if decision == "generated" else "score_label"
    candidate_map = data.predictions[candidate_id]
    golds = [key[2] for key in data.keys]
    preds = [candidate_map[key][label_key] for key in data.keys]
    return golds, preds


def evaluate_candidate(data, candidate_id, decision="generated"):
    golds, preds = candidate_vector(data, candidate_id, decision)
    return compute_metrics(golds, preds)


def select_best_candidate_on_dev(data, ids):
    best_id = ids[0]
    best_decision = "generated"
    best_metrics = evaluate_candidate(data, best_id, best_decision)
    best_tuple = (best_metrics["accuracy"], best_metrics["macro_f1"])

    for candidate_id in ids:
        for decision in ["generated", "score"]:
            metrics = evaluate_candidate(data, candidate_id, decision)
            score_tuple = (metrics["accuracy"], metrics["macro_f1"])
            if score_tuple > best_tuple:
                best_id = candidate_id
                best_decision = decision
                best_tuple = score_tuple

    return best_id, best_decision


def majority_vote_one(labels, margins):
    """Ties are broken by the largest score margin, then by label order."""
    counts = Counter(labels)
    top_count = max(counts.values())
    tied_labels = [label for label in LABELS if counts.get(label, 0) == top_count]

    if len(tied_labels) == 1:
        label = tied_labels[0]
        label_margins = [m for vote, m in zip(labels, margins) if vote == label]
        return label, False, top_count, max(label_margins) if label_margins else 0.0

    best_label = tied_labels[0]
    best_score = (-1.0, -999)
    for label in tied_labels:
        label_margins = [m for vote, m in zip(labels, margins) if vote == label]
        max_margin = max(label_margins) if label_margins else 0.0
        score = (max_margin, -LABEL_TO_ID[label])
        if score > best_score:
            best_label = label
            best_score = score

    return best_label, True, top_count, float(best_score[0])


def majority_vote_predictions(data, ids):
    preds, details = [], []

    for key in data.keys:
        labels, margins = [], []
        for candidate_id in ids:
            record = data.predictions[candidate_id][key]
            labels.append(record["generated_label"])
            margins.append(float(record.get("margin", 0.0)))

        pred, tie_used, top_vote_count, tie_break_margin = majority_vote_one(labels, margins)
        preds.append(pred)
        details.append({
            "Flickr30K_ID": key[0],
            "hypothesis": key[1],
            "gold": key[2],
            "prediction": pred,
            "tie_used": int(tie_used),
            "top_vote_count": top_vote_count,
            "tie_break_margin": tie_break_margin,
            "votes_entailment": labels.count("entailment"),
            "votes_neutral": labels.count("neutral"),
            "votes_contradiction": labels.count("contradiction"),
        })

    return preds, details


def oracle_predictions(data, ids, fallback_preds):
    preds = []
    for idx, key in enumerate(data.keys):
        gold = key[2]
        candidate_labels = [data.predictions[cid][key]["generated_label"] for cid in ids]
        preds.append(gold if gold in candidate_labels else fallback_preds[idx])
    return preds


def selected_config(candidate_id, specs_by_id, decision=None):
    spec = specs_by_id[candidate_id]
    text = f"{spec.vlm_display}, {spec.method_display}, {spec.prompt}"
    if decision is not None:
        text += ", " + ("generated" if decision == "generated" else "score-based")
    return text


def main_result_row(method, dev_metrics, test_metrics, selected=""):
    return {
        "Method": method,
        "Selected configuration": selected,
        "Dev Acc": fmt_num(dev_metrics["accuracy"]),
        "Dev Macro-F1": fmt_num(dev_metrics["macro_f1"]),
        "Test Acc": fmt_num(test_metrics["accuracy"]),
        "Test Macro-F1": fmt_num(test_metrics["macro_f1"]),
    }


def save_confusion_csv(path, matrix):
    rows = []
    for i, gold in enumerate(LABELS):
        row = {"gold": gold}
        for j, pred in enumerate(LABELS):
            row[f"pred_{pred}"] = int(matrix[i, j])
        rows.append(row)
    write_csv(path, rows)


def plot_confusion_matrices(dev_matrix, test_matrix, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, title in zip(axes,
                                 [row_normalize(dev_matrix), row_normalize(test_matrix)],
                                 ["Development", "Test"]):
        im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=45, ha="right")
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(f"AVE-MV inter, {title}")
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, "majority_voting_confusion.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "majority_voting_confusion.pdf"))
    plt.close(fig)


def run(out_dir, make_plots=True):
    ensure_dir(out_dir)

    print(SEP, flush=True)
    print("MAJORITY VOTING SUMMARY", flush=True)
    print(SEP, flush=True)
    print(f"Output folder: {out_dir}", flush=True)

    missing = cand.check_required_files(["dev", "test"])
    if missing:
        print("\nMissing required files:", file=sys.stderr, flush=True)
        for path in missing:
            print(f"  {path}", file=sys.stderr, flush=True)
        raise FileNotFoundError(f"Missing {len(missing)} required input files.")

    dev = cand.load_split("dev")
    test = cand.load_split("test")

    specs_by_id = {spec.candidate_id: spec for spec in dev.specs}
    all_ids = cand.candidate_ids(dev.specs)
    qwen_ids = cand.candidate_ids(dev.specs, vlm="qwen")
    internvl_ids = cand.candidate_ids(dev.specs, vlm="internvl")
    full_ids = cand.candidate_ids(dev.specs, method_families=["full_hypothesis"])
    atomic_ids = cand.candidate_ids(
        dev.specs, method_families=["atomic_prediction", "self_decomposed_atomic"])

    best_full_id, best_full_decision = select_best_candidate_on_dev(dev, full_ids)
    best_atomic_id, best_atomic_decision = select_best_candidate_on_dev(dev, atomic_ids)

    gold_dev = [key[2] for key in dev.keys]
    gold_test = [key[2] for key in test.keys]

    mv_qwen_dev, _ = majority_vote_predictions(dev, qwen_ids)
    mv_qwen_test, _ = majority_vote_predictions(test, qwen_ids)
    mv_internvl_dev, _ = majority_vote_predictions(dev, internvl_ids)
    mv_internvl_test, _ = majority_vote_predictions(test, internvl_ids)
    mv_inter_dev, mv_inter_dev_details = majority_vote_predictions(dev, all_ids)
    mv_inter_test, mv_inter_test_details = majority_vote_predictions(test, all_ids)

    oracle_dev = oracle_predictions(dev, all_ids, mv_inter_dev)
    oracle_test = oracle_predictions(test, all_ids, mv_inter_test)

    main_rows = [
        main_result_row("Best full-hypothesis candidate",
                        evaluate_candidate(dev, best_full_id, best_full_decision),
                        evaluate_candidate(test, best_full_id, best_full_decision),
                        selected_config(best_full_id, specs_by_id, best_full_decision)),
        main_result_row("Best atomic candidate",
                        evaluate_candidate(dev, best_atomic_id, best_atomic_decision),
                        evaluate_candidate(test, best_atomic_id, best_atomic_decision),
                        selected_config(best_atomic_id, specs_by_id, best_atomic_decision)),
        main_result_row("AVE-MV intra (Qwen)",
                        compute_metrics(gold_dev, mv_qwen_dev),
                        compute_metrics(gold_test, mv_qwen_test),
                        "Qwen retained pool, K=6"),
        main_result_row("AVE-MV intra (InternVL)",
                        compute_metrics(gold_dev, mv_internvl_dev),
                        compute_metrics(gold_test, mv_internvl_test),
                        "InternVL retained pool, K=6"),
        main_result_row("AVE-MV inter",
                        compute_metrics(gold_dev, mv_inter_dev),
                        compute_metrics(gold_test, mv_inter_test),
                        "Full retained pool, K=12"),
        main_result_row("Oracle upper bound (K=12)",
                        compute_metrics(gold_dev, oracle_dev),
                        compute_metrics(gold_test, oracle_test),
                        "Full retained pool, K=12"),
    ]

    columns = ["Method", "Selected configuration", "Dev Acc", "Dev Macro-F1",
               "Test Acc", "Test Macro-F1"]
    print("\n" + DASH, flush=True)
    print(text_table(main_rows, columns), flush=True)
    print(DASH, flush=True)

    write_csv(os.path.join(out_dir, "majority_voting_main_results.csv"),
              main_rows, columns)

    individual_rows = []
    for spec in dev.specs:
        for decision in ["generated", "score"]:
            dm = evaluate_candidate(dev, spec.candidate_id, decision)
            tm = evaluate_candidate(test, spec.candidate_id, decision)
            individual_rows.append({
                "candidate_id": spec.candidate_id,
                "vlm": spec.vlm_display,
                "method": spec.method_display,
                "prompt": spec.prompt,
                "decision": decision,
                "dev_accuracy": fmt_num(dm["accuracy"]),
                "dev_macro_f1": fmt_num(dm["macro_f1"]),
                "test_accuracy": fmt_num(tm["accuracy"]),
                "test_macro_f1": fmt_num(tm["macro_f1"]),
            })
    write_csv(os.path.join(out_dir, "individual_candidate_results.csv"), individual_rows)

    dev_matrix = confusion_matrix_counts(gold_dev, mv_inter_dev)
    test_matrix = confusion_matrix_counts(gold_test, mv_inter_test)
    save_confusion_csv(os.path.join(out_dir, "confusion_dev.csv"), dev_matrix)
    save_confusion_csv(os.path.join(out_dir, "confusion_test.csv"), test_matrix)

    write_jsonl(os.path.join(out_dir, "ave_mv_inter_dev_predictions.jsonl"),
                mv_inter_dev_details)
    write_jsonl(os.path.join(out_dir, "ave_mv_inter_test_predictions.jsonl"),
                mv_inter_test_details)

    if make_plots:
        plot_confusion_matrices(dev_matrix, test_matrix, out_dir)

    print(f"\nWritten to {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",
                    default=os.path.join(config.OUTPUT_DIR, "majority_voting_summary"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    run(args.out_dir, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()