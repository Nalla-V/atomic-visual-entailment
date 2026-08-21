#!/usr/bin/env python3
"""
Create final AVE-LS result outputs without rerunning evaluation.

Outputs:
1. Compact final performance table as TXT
2. Per-class recall grouped bar chart for dev and test
3. Paired McNemar table for AVE-MV_inter vs AVE-LS

This script reads already generated outputs from:
- AVE_learned_selection_evaluation_v3
- majority voting summary
- Candidate Prediction Analysis / Label-Level Behavior
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(os.environ.get("DATA_ROOT", "."))

AVE_LS_EVAL_NAME = "AVE_learned_selection_evaluation_v3"

MV_DIR = BASE_DIR / "Output" / "majority voting summary"

CANDIDATE_LABEL_DIR = (
    BASE_DIR
    / "Output"
    / "Candidate Prediction Analysis"
    / "Label-Level Behavior"
)

OUT_DIR = BASE_DIR / "Output" / "AVE_learned_selection_summary_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Constants
# ============================================================

LABELS = ["entailment", "neutral", "contradiction"]
DISPLAY_LABELS = ["Entailment", "Neutral", "Contradiction"]

PLOT_BLUE = "#4F81BD"
PLOT_WINE = "#C0504D"
PLOT_GREEN = "#9BBB59"

METHOD_COLORS = {
    "Best individual": PLOT_BLUE,
    "AVE-MV_inter": PLOT_WINE,
    "AVE-LS": PLOT_GREEN,
}

METHOD_ORDER_RECALL = [
    "Best individual",
    "AVE-MV_inter",
    "AVE-LS",
]

METHOD_DISPLAY_NAMES = {
    "Best individual": "Best individual",
    "AVE-MV_inter": r"AVE-MV$_{\mathrm{inter}}$",
    "AVE-LS": "AVE-LS",
}

STRICT_ROW_KEY_CHECK = True


# ============================================================
# Basic helpers
# ============================================================

def safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def fmt3(x: Any) -> str:
    return f"{float(x):.3f}"


def fmt6(x: Any) -> str:
    return f"{float(x):.6g}"


def normalize_label(x: Any) -> str:
    text = safe_text(x).lower()

    if text in LABELS:
        return text

    if "entailment" in text or "entailed" in text or "support" in text:
        return "entailment"

    if "contradiction" in text or "contradict" in text or "conflict" in text:
        return "contradiction"

    if "neutral" in text or "uncertain" in text or "insufficient" in text:
        return "neutral"

    return "neutral"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def text_table(df: pd.DataFrame) -> str:
    str_df = df.astype(str)
    widths = {
        col: max(len(col), str_df[col].map(len).max())
        for col in str_df.columns
    }

    header = " | ".join(col.ljust(widths[col]) for col in str_df.columns)
    sep = "-+-".join("-" * widths[col] for col in str_df.columns)

    body = []
    for _, row in str_df.iterrows():
        body.append(
            " | ".join(str(row[col]).ljust(widths[col]) for col in str_df.columns)
        )

    return "\n".join([header, sep] + body)


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        f"None of these columns were found: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# Input paths
# ============================================================

def ls_eval_dir(split: str) -> Path:
    return (
        BASE_DIR
        / "Output"
        / f"{split}_dataset"
        / AVE_LS_EVAL_NAME
    )


def ls_summary_path(split: str) -> Path:
    return ls_eval_dir(split) / f"ave_ls_v3_{split}_summary.csv"


def ls_predictions_path(split: str) -> Path:
    return ls_eval_dir(split) / f"ave_ls_v3_{split}_predictions.jsonl"


def mv_predictions_path(split: str) -> Path:
    return MV_DIR / f"ave_mv_predictions_{split}.csv"


# ============================================================
# 1. Compact final performance TXT table
# ============================================================

def load_ls_summary(split: str) -> pd.Series:
    path = ls_summary_path(split)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty summary file: {path}")

    return df.iloc[0]


def build_compact_performance_table() -> pd.DataFrame:
    dev_summary = load_ls_summary("dev")
    test_summary = load_ls_summary("test")

    rows = [
        {
            "Method": "Best full-hypothesis",
            "Dev Accuracy": "0.749",
            "Dev Macro-F1": "0.732",
            "Test Accuracy": "0.754",
            "Test Macro-F1": "0.737",
        },
        {
            "Method": "Best atomic method",
            "Dev Accuracy": "0.762",
            "Dev Macro-F1": "0.757",
            "Test Accuracy": "0.765",
            "Test Macro-F1": "0.760",
        },
        {
            "Method": "AVE-MV_inter",
            "Dev Accuracy": "0.765",
            "Dev Macro-F1": "0.756",
            "Test Accuracy": "0.769",
            "Test Macro-F1": "0.760",
        },
        {
            "Method": "AVE-LS",
            "Dev Accuracy": fmt3(dev_summary["accuracy"]),
            "Dev Macro-F1": fmt3(dev_summary["macro_f1"]),
            "Test Accuracy": fmt3(test_summary["accuracy"]),
            "Test Macro-F1": fmt3(test_summary["macro_f1"]),
        },
        {
            "Method": "Oracle (K=12)",
            "Dev Accuracy": "0.941",
            "Dev Macro-F1": "0.941",
            "Test Accuracy": "0.941",
            "Test Macro-F1": "0.940",
        },
    ]

    out = pd.DataFrame(rows)

    out_csv = OUT_DIR / "ave_ls_final_compact_performance_table.csv"
    out_txt = OUT_DIR / "ave_ls_final_compact_performance_table.txt"

    out.to_csv(out_csv, index=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Final AVE performance compact table\n")
        f.write("=" * 80 + "\n")
        f.write(text_table(out))
        f.write("\n")

    return out


# ============================================================
# 2. Per-class recall grouped bar chart
# ============================================================

def load_best_individual_recall() -> pd.DataFrame:
    path = CANDIDATE_LABEL_DIR / "best_individual_confusion_matrices.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing best individual confusion CSV: {path}\n"
            "This file should already exist from candidate prediction analysis."
        )

    df = pd.read_csv(path)

    rows = []
    for split in ["dev", "test"]:
        for label in LABELS:
            sub = df[
                (df["split"] == split)
                & (df["gold_label"] == label)
                & (df["predicted_label"] == label)
            ]

            if sub.empty:
                raise ValueError(f"Missing diagonal recall for {split}, {label}")

            rows.append({
                "split": split,
                "method": "Best individual",
                "label": label,
                "recall": float(sub.iloc[0]["row_normalized"]),
            })

    return pd.DataFrame(rows)


def load_mv_inter_recall() -> pd.DataFrame:
    rows = []

    for split in ["dev", "test"]:
        path = MV_DIR / f"confusion_matrix_ave_mv_inter_{split}.csv"
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        for label in LABELS:
            sub = df[
                (df["gold_label"] == label)
                & (df["predicted_label"] == label)
            ]

            if sub.empty:
                raise ValueError(f"Missing MV diagonal recall for {split}, {label}")

            rows.append({
                "split": split,
                "method": "AVE-MV_inter",
                "label": label,
                "recall": float(sub.iloc[0]["row_normalized"]),
            })

    return pd.DataFrame(rows)


def load_ave_ls_recall() -> pd.DataFrame:
    rows = []

    for split in ["dev", "test"]:
        summary = load_ls_summary(split)

        for label in LABELS:
            rows.append({
                "split": split,
                "method": "AVE-LS",
                "label": label,
                "recall": float(summary[f"{label}_recall"]),
            })

    return pd.DataFrame(rows)


def build_recall_dataframe() -> pd.DataFrame:
    recall_df = pd.concat(
        [
            load_best_individual_recall(),
            load_mv_inter_recall(),
            load_ave_ls_recall(),
        ],
        ignore_index=True,
    )

    recall_df.to_csv(OUT_DIR / "ave_ls_recall_comparison_dev_test.csv", index=False)
    return recall_df


def plot_recall_grouped_bar(recall_df: pd.DataFrame) -> None:
    x = np.arange(len(LABELS))
    width = 0.23

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)

    panels = [
        ("dev", "Development"),
        ("test", "Test"),
    ]

    for ax_idx, (ax, (split, title)) in enumerate(zip(axes, panels)):
        split_df = recall_df[recall_df["split"] == split]

        for method_idx, method in enumerate(METHOD_ORDER_RECALL):
            values = []

            for label in LABELS:
                sub = split_df[
                    (split_df["method"] == method)
                    & (split_df["label"] == label)
                ]

                if sub.empty:
                    raise ValueError(f"Missing recall for {split}, {method}, {label}")

                values.append(float(sub.iloc[0]["recall"]))

            offset = (method_idx - 1) * width

            bars = ax.bar(
                x + offset,
                values,
                width=width,
                label=METHOD_DISPLAY_NAMES[method],
                color=METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.8,
            )

            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        ax.set_title(title, fontsize=15, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(DISPLAY_LABELS, fontsize=15)
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", alpha=0.35, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=12)

        if ax_idx == 0:
            ax.set_ylabel("Recall", fontsize=15)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        fontsize=15,
        bbox_to_anchor=(0.5, -0.005),
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    fig.savefig(
        OUT_DIR / "ave_ls_recall_comparison_grouped_bar_dev_test.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / "ave_ls_recall_comparison_grouped_bar_dev_test.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# 3. Paired McNemar table
# ============================================================

def add_row_alignment_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds row_index and normalized keys.

    McNemar is computed by row_index alignment, because the same image-hypothesis
    key can occur more than once. The normalized keys are used as a safety check
    after row-wise alignment.
    """
    df = df.copy()

    df["row_index"] = np.arange(len(df), dtype=int)
    df["Flickr30K_ID_norm"] = df["Flickr30K_ID"].map(safe_text)
    df["hypothesis_norm"] = df["hypothesis"].map(safe_text)
    df["gold_norm"] = df["gold"].map(normalize_label)

    return df


def load_ls_predictions_for_mcnemar(split: str) -> pd.DataFrame:
    path = ls_predictions_path(split)
    if not path.exists():
        raise FileNotFoundError(path)

    rows = read_jsonl(path)

    keep_rows = []
    for row in rows:
        gold = normalize_label(row.get("gold", row.get("annotator_label", "")))

        pred = normalize_label(
            row.get(
                "final_label",
                row.get(
                    "prediction",
                    row.get(
                        "learned_label",
                        row.get("predicted_label", ""),
                    ),
                ),
            )
        )

        keep_rows.append({
            "Flickr30K_ID": safe_text(row.get("Flickr30K_ID", "")),
            "hypothesis": safe_text(row.get("hypothesis", "")),
            "gold": gold,
            "ave_ls": pred,
            "ave_ls_correct": int(pred == gold),
        })

    return add_row_alignment_metadata(pd.DataFrame(keep_rows))


def load_mv_predictions_for_mcnemar(split: str) -> pd.DataFrame:
    path = mv_predictions_path(split)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    id_col = first_existing_column(
        df,
        ["Flickr30K_ID", "flickr30k_id", "image_id", "image_id_raw"],
    )
    hyp_col = first_existing_column(
        df,
        ["hypothesis", "Hypothesis"],
    )
    gold_col = first_existing_column(
        df,
        ["gold", "gold_label", "annotator_label", "label"],
    )
    pred_col = first_existing_column(
        df,
        ["ave_mv_inter", "AVE-MV_inter", "mv_inter", "prediction", "predicted_label"],
    )

    keep = pd.DataFrame({
        "Flickr30K_ID": df[id_col].map(safe_text),
        "hypothesis": df[hyp_col].map(safe_text),
        "gold": df[gold_col].map(normalize_label),
        "ave_mv_inter": df[pred_col].map(normalize_label),
    })

    keep["ave_mv_inter_correct"] = (
        keep["ave_mv_inter"] == keep["gold"]
    ).astype(int)

    return add_row_alignment_metadata(keep)

def validate_row_alignment(merged: pd.DataFrame, split: str) -> None:
    """
    Checks that row-wise alignment also matches the existing keys.

    If this fails, the two files are not in the same example order.
    """
    checks = {
        "Flickr30K_ID": (
            merged["Flickr30K_ID_norm_mv"] == merged["Flickr30K_ID_norm_ls"]
        ),
        "hypothesis": (
            merged["hypothesis_norm_mv"] == merged["hypothesis_norm_ls"]
        ),
        "gold": (
            merged["gold_norm_mv"] == merged["gold_norm_ls"]
        ),
    }

    mismatch_mask = ~(
        checks["Flickr30K_ID"]
        & checks["hypothesis"]
        & checks["gold"]
    )

    mismatch_count = int(mismatch_mask.sum())

    if mismatch_count == 0:
        print(f"{split}: row-index alignment passed key check.")
        return

    mismatch_path = OUT_DIR / f"row_alignment_mismatches_{split}.csv"
    merged.loc[
        mismatch_mask,
        [
            "row_index",
            "Flickr30K_ID_norm_mv",
            "Flickr30K_ID_norm_ls",
            "hypothesis_norm_mv",
            "hypothesis_norm_ls",
            "gold_norm_mv",
            "gold_norm_ls",
        ],
    ].to_csv(mismatch_path, index=False)

    message = (
        f"{split}: row-index alignment has {mismatch_count} key mismatches. "
        f"Saved mismatch audit to {mismatch_path}"
    )

    if STRICT_ROW_KEY_CHECK:
        raise ValueError(message)

    print("WARNING:", message)


def exact_two_sided_binomial_p(k: int, n: int) -> float:
    """
    Exact two-sided binomial p-value with p=0.5.

    This matches the exact McNemar binomial test on the discordant pairs.
    """
    if n == 0:
        return 1.0

    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n=n, p=0.5, alternative="two-sided").pvalue)
    except Exception:
        probs = [math.comb(n, i) * (0.5 ** n) for i in range(n + 1)]
        observed = probs[k]
        return float(sum(p for p in probs if p <= observed + 1e-15))


def compute_mcnemar(split: str) -> Dict[str, Any]:
    mv_df = load_mv_predictions_for_mcnemar(split)
    ls_df = load_ls_predictions_for_mcnemar(split)

    if len(mv_df) != len(ls_df):
        print(
            f"WARNING: {split} file lengths differ. "
            f"MV={len(mv_df)}, LS={len(ls_df)}. "
            "McNemar will use the common row_index values only."
        )

    merged = mv_df.merge(
        ls_df,
        on="row_index",
        suffixes=("_mv", "_ls"),
        how="inner",
    )

    validate_row_alignment(merged, split)

    mv_correct = merged["ave_mv_inter_correct"] == 1
    ls_correct = merged["ave_ls_correct"] == 1

    mv_correct_ls_wrong = int((mv_correct & ~ls_correct).sum())
    ls_correct_mv_wrong = int((ls_correct & ~mv_correct).sum())

    discordant = mv_correct_ls_wrong + ls_correct_mv_wrong
    net = ls_correct_mv_wrong - mv_correct_ls_wrong

    p_value = exact_two_sided_binomial_p(
        k=ls_correct_mv_wrong,
        n=discordant,
    )

    return {
        "Split": "Development" if split == "dev" else "Test",
        "N matched": int(len(merged)),
        "AVE-MV_inter correct, AVE-LS wrong": mv_correct_ls_wrong,
        "AVE-LS correct, AVE-MV_inter wrong": ls_correct_mv_wrong,
        "Net": net,
        "Two-sided exact binomial p": p_value,
    }


def build_mcnemar_table() -> pd.DataFrame:
    rows = [compute_mcnemar("dev"), compute_mcnemar("test")]
    df = pd.DataFrame(rows)

    df.to_csv(OUT_DIR / "ave_ls_vs_ave_mv_inter_mcnemar.csv", index=False)

    display_df = df.copy()
    display_df["Two-sided exact binomial p"] = display_df[
        "Two-sided exact binomial p"
    ].map(fmt6)

    with open(OUT_DIR / "ave_ls_vs_ave_mv_inter_mcnemar.txt", "w", encoding="utf-8") as f:
        f.write("Paired McNemar test: AVE-MV_inter vs AVE-LS\n")
        f.write("=" * 90 + "\n")
        f.write(text_table(display_df))
        f.write("\n")

    return df


# ============================================================
# Main
# ============================================================

def main() -> None:
    compact_df = build_compact_performance_table()
    recall_df = build_recall_dataframe()
    plot_recall_grouped_bar(recall_df)
    mcnemar_df = build_mcnemar_table()

    print("\nSaved outputs to:")
    print(f"  {OUT_DIR}")

    print("\nCompact performance table:")
    print(text_table(compact_df))

    print("\nMcNemar table:")
    display_mcnemar = mcnemar_df.copy()
    display_mcnemar["Two-sided exact binomial p"] = display_mcnemar[
        "Two-sided exact binomial p"
    ].map(fmt6)
    print(text_table(display_mcnemar))


if __name__ == "__main__":
    main()