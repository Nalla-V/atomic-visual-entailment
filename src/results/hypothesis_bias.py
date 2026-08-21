import os
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================

BASE_DIR = os.environ.get("DATA_ROOT", ".")
TEST_DATASET_DIR = os.path.join(BASE_DIR, "Output/test_dataset")

REAL_DIRS = {
    "qwen3": os.path.join(TEST_DATASET_DIR, "qwen3_predictions_clean_v2"),
    "internvl": os.path.join(TEST_DATASET_DIR, "internvl_predictions_clean_v2"),
}

BIAS_DIRS = {
    "qwen3_black": os.path.join(TEST_DATASET_DIR, "hypothesis_bias/qwen3_black"),
    "qwen3_white": os.path.join(TEST_DATASET_DIR, "hypothesis_bias/qwen3_white"),
    "internvl_black": os.path.join(TEST_DATASET_DIR, "hypothesis_bias/internvl_black"),
    "internvl_white": os.path.join(TEST_DATASET_DIR, "hypothesis_bias/internvl_white"),
}

OUTPUT_DIR = os.path.join(TEST_DATASET_DIR, "hypothesis_bias", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "hypothesis_bias_accuracy_summary.csv"
)

BIAS_DELTA_CSV = os.path.join(
    OUTPUT_DIR,
    "hypothesis_bias_delta_accuracy.csv"
)

BIAS_DELTA_TXT = os.path.join(
    OUTPUT_DIR,
    "hypothesis_bias_delta_accuracy.txt"
)

PLOT_PNG = os.path.join(
    OUTPUT_DIR,
    "hypothesis_bias_grouped_bar_test.png"
)

PLOT_PDF = os.path.join(
    OUTPUT_DIR,
    "hypothesis_bias_grouped_bar_test.pdf"
)

LABELS = ["entailment", "neutral", "contradiction"]

# Which label output to use.
# Options: "prediction" or "score_prediction".
PREDICTION_FIELD = "prediction"

# Plot colours
PLOT_BLUE = "#4F81BD"
PLOT_WINE = "#C0504D"
PLOT_GREEN = "#9BBB59"

# Random baseline for three labels
RANDOM_BASELINE = 1.0 / 3.0

# Figure and font sizes. Adjust these if needed.
FIGSIZE = (10.8, 5.8)
Y_LABEL_FONTSIZE = 15
X_TICK_FONTSIZE = 15
Y_TICK_FONTSIZE = 13
BAR_VALUE_FONTSIZE = 10.5
LEGEND_FONTSIZE = 15
RANDOM_LABEL_FONTSIZE = 11.5

# Bar and axis settings
BAR_WIDTH = 0.23
BAR_VALUE_OFFSET = 0.010
Y_MIN = 0.25
Y_MAX = 0.82
X_LEFT_PAD = 0.55
X_RIGHT_PAD = 0.82

# Grid and spine settings
GRID_ALPHA = 0.18
GRID_LINEWIDTH = 0.7
SPINE_LINEWIDTH = 0.9

# Random baseline styling: visible, but less dominant than black
RANDOM_LINE_COLOR = "#4A4A4A"
RANDOM_LINEWIDTH = 1.15
RANDOM_LINE_ALPHA = 0.95
RANDOM_LINE_DASHES = (5, 3)
RANDOM_LABEL_TEXT = "0.333"
RANDOM_LABEL_Y_OFFSET = 0.006
RANDOM_LABEL_X_OFFSET = 0.18


METHODS = [
    {
        "display_name": "Qwen3\nFull-hypothesis",
        "model": "qwen3",
        "method_type": "baseline",
        "filename": "baseline_structured_test.jsonl",
        "real_dir_key": "qwen3",
        "black_dir_key": "qwen3_black",
        "white_dir_key": "qwen3_white",
    },
    {
        "display_name": "Qwen3\nAtomic prediction",
        "model": "qwen3",
        "method_type": "joint_atomic",
        "filename": "atomic_joint_structured_test.jsonl",
        "real_dir_key": "qwen3",
        "black_dir_key": "qwen3_black",
        "white_dir_key": "qwen3_white",
    },
    {
        "display_name": "InternVL3\nFull-hypothesis",
        "model": "internvl",
        "method_type": "baseline",
        "filename": "baseline_structured_test.jsonl",
        "real_dir_key": "internvl",
        "black_dir_key": "internvl_black",
        "white_dir_key": "internvl_white",
    },
    {
        "display_name": "InternVL3\nAtomic prediction",
        "model": "internvl",
        "method_type": "joint_atomic",
        "filename": "atomic_joint_structured_test.jsonl",
        "real_dir_key": "internvl",
        "black_dir_key": "internvl_black",
        "white_dir_key": "internvl_white",
    },
]


# ============================================================
# 2. HELPERS
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

    if "contradiction" in text or "contradict" in text or "conflict" in text:
        return "contradiction"

    if "neutral" in text or "uncertain" in text or "unknown" in text or "insufficient" in text:
        return "neutral"

    return ""


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Could not parse line {line_no} in {path}: {e}")

    return rows


def get_result_block(row: Dict[str, Any], method_type: str) -> Dict[str, Any]:
    if method_type == "baseline":
        return row.get("full_hypothesis_results", {}) or {}

    if method_type == "joint_atomic":
        return row.get("joint_atom_results", {}) or {}

    raise ValueError(f"Unknown method_type: {method_type}")


def extract_prediction(row: Dict[str, Any], method_type: str) -> str:
    result_block = get_result_block(row, method_type)

    pred = normalize_label(result_block.get(PREDICTION_FIELD, ""))

    if not pred:
        pred = normalize_label(result_block.get("prediction", ""))

    if not pred:
        pred = normalize_label(result_block.get("score_prediction", ""))

    if not pred:
        pred = normalize_label(result_block.get("top_label", ""))

    return pred


def extract_gold(row: Dict[str, Any]) -> str:
    return normalize_label(
        row.get(
            "annotator_label",
            row.get("gold", row.get("label", ""))
        )
    )


# ============================================================
# 3. ACCURACY PER FILE
# ============================================================

def evaluate_file(path: str, method_type: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    rows = read_jsonl(path)

    total = 0
    correct = 0
    skipped = 0

    y_true = []
    y_pred = []

    for row in rows:
        gold = extract_gold(row)
        pred = extract_prediction(row, method_type)

        if gold not in LABELS or pred not in LABELS:
            skipped += 1
            continue

        total += 1
        correct += int(gold == pred)

        y_true.append(gold)
        y_pred.append(pred)

    if total == 0:
        raise RuntimeError(f"No valid rows found in file: {path}")

    accuracy = correct / total

    return {
        "path": path,
        "n_total_file_rows": len(rows),
        "n_used": total,
        "n_skipped": skipped,
        "correct": correct,
        "accuracy": accuracy,
        "gold_distribution": dict(pd.Series(y_true).value_counts()),
        "prediction_distribution": dict(pd.Series(y_pred).value_counts()),
    }


# ============================================================
# 4. BUILD SUMMARY
# ============================================================

def build_summary() -> pd.DataFrame:
    summary_rows = []

    for item in METHODS:
        display_name = item["display_name"]
        method_type = item["method_type"]
        filename = item["filename"]

        real_path = os.path.join(REAL_DIRS[item["real_dir_key"]], filename)
        black_path = os.path.join(BIAS_DIRS[item["black_dir_key"]], filename)
        white_path = os.path.join(BIAS_DIRS[item["white_dir_key"]], filename)

        print("=" * 120)
        print(display_name.replace("\n", " "))
        print("=" * 120)
        print(f"Real : {real_path}")
        print(f"Black: {black_path}")
        print(f"White: {white_path}")

        real_res = evaluate_file(real_path, method_type)
        black_res = evaluate_file(black_path, method_type)
        white_res = evaluate_file(white_path, method_type)

        delta_black = real_res["accuracy"] - black_res["accuracy"]
        delta_white = real_res["accuracy"] - white_res["accuracy"]

        summary_rows.append({
            "method_display": display_name,
            "method_display_clean": display_name.replace("\n", " "),
            "model": item["model"],
            "method_type": method_type,
            "filename": filename,
            "prediction_field_used": PREDICTION_FIELD,

            "real_accuracy": real_res["accuracy"],
            "black_accuracy": black_res["accuracy"],
            "white_accuracy": white_res["accuracy"],

            "delta_black_accuracy": delta_black,
            "delta_white_accuracy": delta_white,

            "real_n": real_res["n_used"],
            "black_n": black_res["n_used"],
            "white_n": white_res["n_used"],

            "real_correct": real_res["correct"],
            "black_correct": black_res["correct"],
            "white_correct": white_res["correct"],

            "real_skipped": real_res["n_skipped"],
            "black_skipped": black_res["n_skipped"],
            "white_skipped": white_res["n_skipped"],

            "real_path": real_path,
            "black_path": black_path,
            "white_path": white_path,
        })

        print(
            f"Real={real_res['accuracy']:.3f} | "
            f"Black={black_res['accuracy']:.3f} | "
            f"White={white_res['accuracy']:.3f} | "
            f"Delta_black={delta_black:.3f} | "
            f"Delta_white={delta_white:.3f}"
        )
        print("")

    return pd.DataFrame(summary_rows)


# ============================================================
# 5. SAVE BIAS DELTA SUMMARY
# ============================================================

def save_bias_delta_outputs(df: pd.DataFrame) -> None:
    delta_cols = [
        "method_display_clean",
        "real_accuracy",
        "black_accuracy",
        "white_accuracy",
        "delta_black_accuracy",
        "delta_white_accuracy",
        "real_n",
        "black_n",
        "white_n",
    ]

    delta_df = df[delta_cols].copy()
    delta_df.to_csv(BIAS_DELTA_CSV, index=False)

    lines = []
    lines.append("Bias delta accuracy summary")
    lines.append("=" * 110)
    lines.append(
        f"{'Method':35s} | {'Real':>8s} | {'Black':>8s} | {'White':>8s} | "
        f"{'Delta black':>12s} | {'Delta white':>12s}"
    )
    lines.append("-" * 110)

    for _, row in delta_df.iterrows():
        method_name = str(row["method_display_clean"])[:35]

        lines.append(
            f"{method_name:35s} | "
            f"{float(row['real_accuracy']):8.3f} | "
            f"{float(row['black_accuracy']):8.3f} | "
            f"{float(row['white_accuracy']):8.3f} | "
            f"{float(row['delta_black_accuracy']):12.3f} | "
            f"{float(row['delta_white_accuracy']):12.3f}"
        )

    with open(BIAS_DELTA_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved bias delta CSV: {BIAS_DELTA_CSV}")
    print(f"Saved bias delta TXT: {BIAS_DELTA_TXT}")


# ============================================================
# 6. GROUPED BAR CHART
# ============================================================

def make_grouped_bar_chart(df: pd.DataFrame) -> None:
    plot_df = df.copy()

    order = [m["display_name"] for m in METHODS]
    plot_df["order"] = plot_df["method_display"].apply(lambda x: order.index(x))
    plot_df = plot_df.sort_values("order")

    methods = plot_df["method_display"].tolist()

    real_values = plot_df["real_accuracy"].to_numpy()
    black_values = plot_df["black_accuracy"].to_numpy()
    white_values = plot_df["white_accuracy"].to_numpy()

    x = np.arange(len(methods))
    width = BAR_WIDTH

    fig, ax = plt.subplots(figsize=FIGSIZE)

    bars_real = ax.bar(
        x - width,
        real_values,
        width,
        label="Real image",
        color=PLOT_BLUE,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    bars_black = ax.bar(
        x,
        black_values,
        width,
        label="Black image",
        color=PLOT_WINE,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    bars_white = ax.bar(
        x + width,
        white_values,
        width,
        label="White image",
        color=PLOT_GREEN,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )

    # Random baseline
    random_line = ax.axhline(
        RANDOM_BASELINE,
        linestyle="--",
        linewidth=RANDOM_LINEWIDTH,
        color=RANDOM_LINE_COLOR,
        alpha=RANDOM_LINE_ALPHA,
        zorder=2,
    )
    random_line.set_dashes(RANDOM_LINE_DASHES)

    ax.text(
        x[-1] + width + RANDOM_LABEL_X_OFFSET,
        RANDOM_BASELINE + RANDOM_LABEL_Y_OFFSET,
        RANDOM_LABEL_TEXT,
        ha="left",
        va="bottom",
        fontsize=RANDOM_LABEL_FONTSIZE,
        color=RANDOM_LINE_COLOR,
    )

    # Bar value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + BAR_VALUE_OFFSET,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=BAR_VALUE_FONTSIZE,
                zorder=4,
            )

    add_labels(bars_real)
    add_labels(bars_black)
    add_labels(bars_white)

    ax.set_ylabel("Accuracy", fontsize=Y_LABEL_FONTSIZE)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=X_TICK_FONTSIZE)
    ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(-X_LEFT_PAD, x[-1] + X_RIGHT_PAD)

    ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=GRID_LINEWIDTH, zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(PLOT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {PLOT_PNG}")
    print(f"Saved plot: {PLOT_PDF}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    df = build_summary()

    df.to_csv(SUMMARY_CSV, index=False)

    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)

    display_cols = [
        "method_display_clean",
        "real_accuracy",
        "black_accuracy",
        "white_accuracy",
        "delta_black_accuracy",
        "delta_white_accuracy",
        "real_n",
        "black_n",
        "white_n",
    ]

    print(df[display_cols].to_string(index=False))
    print("")

    save_bias_delta_outputs(df)
    make_grouped_bar_chart(df)

    print(f"Saved summary CSV: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
