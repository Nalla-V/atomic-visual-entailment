"""Compare the VLMs on full-hypothesis prediction.

    python -m src.results.vlm_comparison

Evaluates each VLM's full-hypothesis predictions across both prompt styles and
both label derivations, selects the best configuration per model, and writes a
summary and the comparison figure.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import config

VALID_LABELS = set(config.FINAL_LABELS)

# Ordered small to large, which is how the figure reads left to right.
MODEL_SPECS = [
    {"model_id": "internvl3_1b", "display_name": "InternVL3-1B-hf",
     "folder": "internvl3_1b_predictions_clean_v2",
     "xtick_label": "InternVL3-\n1B", "color": "#fadc39"},
    {"model_id": "qwen2vl_2b", "display_name": "Qwen2-VL-2B-Instruct",
     "folder": "qwen2vl_2b_predictions_clean_v2",
     "xtick_label": "Qwen2-VL-\n2B", "color": "#B3A6C8"},
    {"model_id": "llava_onevision_7b", "display_name": "LLaVA-OneVision-7B",
     "folder": "llava_onevision_predictions_clean_v2",
     "xtick_label": "LLaVA-\nOneVision-7B", "color": "#8AC6BC"},
    {"model_id": "idefics2_8b", "display_name": "Idefics2-8B",
     "folder": "idefics2_predictions_clean_v2",
     "xtick_label": "Idefics2-\n8B", "color": "#F17C6B"},
    {"model_id": "internvl3_8b", "display_name": "InternVL3-8B-hf",
     "folder": "internvl_predictions_clean_v2",
     "xtick_label": "InternVL3-\n8B", "color": "#A8CF63"},
    {"model_id": "qwen3vl_8b", "display_name": "Qwen3-VL-8B-Instruct",
     "folder": "qwen3_predictions_clean_v2",
     "xtick_label": "Qwen3-VL-\n8B", "color": "#7DA6C9"},
    {"model_id": "qwen3vl_32b", "display_name": "Qwen3-VL-32B-Instruct",
     "folder": "qwen3vl_32b_predictions_clean_v2",
     "xtick_label": "Qwen3-VL-\n32B", "color": "#e3be96"},
]

PROMPT_FILES = {
    "simple": "baseline_simple_{split}.jsonl",
    "structured": "baseline_structured_{split}.jsonl",
}

# Figure appearance
TWO_PANEL_FIGSIZE = (8.5, 3.2)
WIDTH_RATIOS = [1.25, 0.75]
WSPACE = 0.25
BAR_WIDTH = 0.78
Y_LABEL_FONTSIZE = 18
X_TICK_FONTSIZE = 13
Y_TICK_FONTSIZE = 14
BAR_VALUE_FONTSIZE = 12
GRID_ALPHA = 0.28
GRID_LINEWIDTH = 0.8
SPINE_LINEWIDTH = 1.1
BAR_TEXT_OFFSET = 0.004

SUMMARY_FILENAME = "vlm_full_hypothesis_{split}_summary.txt"
PLOT_BASENAME = "vlm_full_hypothesis_best_{split}_accuracy"


def normalize_label(x):
    if x is None:
        return ""
    return str(x).strip().lower()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"WARNING: Bad JSON in {path.name} at line {line_no}")
    return rows


def extract_predictions(path):
    """Gold, generated and score-based labels, keeping only rows where all
    three are valid."""
    golds, generated, score_based = [], [], []

    for obj in load_jsonl(path):
        gold = normalize_label(obj.get("annotator_label"))
        res = obj.get("full_hypothesis_results", {})
        pred_generated = normalize_label(res.get("prediction"))
        pred_score = normalize_label(res.get("score_prediction"))

        if gold not in VALID_LABELS:
            continue
        if pred_generated not in VALID_LABELS:
            continue
        if pred_score not in VALID_LABELS:
            continue

        golds.append(gold)
        generated.append(pred_generated)
        score_based.append(pred_score)

    return {"gold": golds, "generated": generated, "score_based": score_based}


def compute_metrics(gold, pred):
    if len(gold) == 0:
        return 0, float("nan"), float("nan")
    acc = accuracy_score(gold, pred)
    macro_f1 = f1_score(gold, pred, labels=config.FINAL_LABELS, average="macro")
    return len(gold), float(acc), float(macro_f1)


def evaluate_one_model(model_spec, dataset_dir, split):
    results = []
    folder = dataset_dir / model_spec["folder"]

    for prompt, template in PROMPT_FILES.items():
        fname = template.format(split=split)
        path = folder / fname

        if not path.exists():
            print(f"WARNING: Missing file -> {path}")
            continue

        extracted = extract_predictions(path)
        gold = extracted["gold"]

        for decision_mode in ["generated", "score_based"]:
            n, acc, macro_f1 = compute_metrics(gold, extracted[decision_mode])
            results.append({
                "model_id": model_spec["model_id"],
                "display_name": model_spec["display_name"],
                "folder": model_spec["folder"],
                "prompt": prompt,
                "decision_mode": decision_mode,
                "n": n,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "file": fname,
            })

    return results


def select_best_config(rows):
    """Highest accuracy, then macro-F1, then simple over structured, then
    generated over score-based, so ties resolve deterministically."""
    prompt_rank = {"simple": 1, "structured": 0}
    decision_rank = {"generated": 1, "score_based": 0}

    return sorted(rows, key=lambda r: (
        r["accuracy"], r["macro_f1"],
        prompt_rank.get(r["prompt"], -1),
        decision_rank.get(r["decision_mode"], -1),
    ), reverse=True)[0]


def make_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "-+-".join("-" * w for w in widths)
    lines = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), sep]
    for row in rows:
        lines.append(" | ".join(str(cell).ljust(widths[i])
                                for i, cell in enumerate(row)))
    return "\n".join(lines)


def write_summary(best_rows, all_rows, out_dir, dataset_dir, split):
    out_path = out_dir / SUMMARY_FILENAME.format(split=split)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"VLM FULL-HYPOTHESIS COMPARISON ({split.upper()} SET)\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Base directory : {dataset_dir}\n")
        f.write(f"Output folder  : {out_dir}\n\n")

        f.write("Best configuration per model\n")
        f.write("-" * 100 + "\n")
        f.write(make_table(
            ["Model", "Best prompt", "Best decision", "Accuracy", "Macro-F1", "N"],
            [[r["display_name"], r["prompt"], r["decision_mode"],
              f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}", str(r["n"])]
             for r in best_rows]))
        f.write("\n\n")

        f.write("Accuracy ranking (best selected configuration only)\n")
        f.write("-" * 100 + "\n")
        ranked = sorted(best_rows, key=lambda r: (r["accuracy"], r["macro_f1"]),
                        reverse=True)
        f.write(make_table(
            ["Rank", "Model", "Prompt", "Decision", "Accuracy", "Macro-F1"],
            [[str(i), r["display_name"], r["prompt"], r["decision_mode"],
              f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}"]
             for i, r in enumerate(ranked, 1)]))
        f.write("\n\n")

        f.write("All evaluated full-hypothesis configurations\n")
        f.write("-" * 100 + "\n")
        f.write(make_table(
            ["Model", "Prompt", "Decision", "Accuracy", "Macro-F1", "N", "File"],
            [[r["display_name"], r["prompt"], r["decision_mode"],
              f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}", str(r["n"]), r["file"]]
             for r in all_rows]))
        f.write("\n")

    print(f"Saved summary -> {out_path}")


def plot_best_accuracy(best_rows, out_dir, split):
    """The bar chart sits in the left panel, with a blank right panel so the
    figure keeps the same proportions as the other two-panel thesis figures."""
    best_map = {row["model_id"]: row for row in best_rows}

    colors, xtick_labels, accuracies = [], [], []
    for spec in MODEL_SPECS:
        row = best_map[spec["model_id"]]
        colors.append(spec["color"])
        xtick_labels.append(spec["xtick_label"])
        accuracies.append(row["accuracy"])

    x = np.arange(len(accuracies))

    fig, axes = plt.subplots(1, 2, figsize=TWO_PANEL_FIGSIZE,
                             gridspec_kw={"width_ratios": WIDTH_RATIOS,
                                          "wspace": WSPACE})
    ax, dummy_ax = axes[0], axes[1]
    dummy_ax.axis("off")

    bars = ax.bar(x, accuracies, width=BAR_WIDTH, color=colors, edgecolor="none")

    ax.set_ylim(max(0.0, min(accuracies) - 0.03), min(1.0, max(accuracies) + 0.03))
    ax.set_ylabel("Accuracy", fontsize=Y_LABEL_FONTSIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=X_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)

    for bar, value in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, value + BAR_TEXT_OFFSET,
                f"{value:.3f}", ha="center", va="bottom",
                fontsize=BAR_VALUE_FONTSIZE)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.22, wspace=WSPACE)

    basename = PLOT_BASENAME.format(split=split)
    png_path = out_dir / f"{basename}.png"
    pdf_path = out_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure -> {png_path}")
    print(f"Saved figure -> {pdf_path}")


def run(split, output_name):
    dataset_dir = Path(config.OUTPUT_DIR) / f"{split}_dataset"
    out_dir = Path(config.OUTPUT_DIR) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("VLM FULL-HYPOTHESIS COMPARISON")
    print("=" * 100)
    print(f"Split : {split}")
    print(f"Input : {dataset_dir}")
    print(f"Output: {out_dir}\n")

    all_rows = []
    for spec in MODEL_SPECS:
        model_rows = evaluate_one_model(spec, dataset_dir, split)
        if not model_rows:
            raise RuntimeError(f"No valid results found for {spec['display_name']}")
        all_rows.extend(model_rows)
        print(f"Loaded {spec['display_name']}: {len(model_rows)} configurations")

    model_order = {spec["model_id"]: i for i, spec in enumerate(MODEL_SPECS)}
    prompt_order = {"simple": 0, "structured": 1}
    decision_order = {"generated": 0, "score_based": 1}

    all_rows.sort(key=lambda r: (model_order[r["model_id"]],
                                 prompt_order[r["prompt"]],
                                 decision_order[r["decision_mode"]]))

    best_rows = [select_best_config([r for r in all_rows
                                     if r["model_id"] == spec["model_id"]])
                 for spec in MODEL_SPECS]

    write_summary(best_rows, all_rows, out_dir, dataset_dir, split)
    plot_best_accuracy(best_rows, out_dir, split)

    print("\nBest configuration per model")
    print("-" * 88)
    for r in best_rows:
        print(f"{r['display_name']:<26} {r['prompt']:<12} {r['decision_mode']:<14} "
              f"acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}")

    print("\nDone.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--output-name", default="vlm_comparison")
    args = ap.parse_args()
    run(args.split, args.output_name)


if __name__ == "__main__":
    main()