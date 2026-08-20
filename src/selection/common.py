"""Shared helpers for the selection stage: keys, labels, scores, metrics."""

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src import config

LABELS = config.FINAL_LABELS
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_label(value):
    text = safe_text(value).lower()
    for label in LABELS:
        if label in text:
            return label
    return ""


def fmt_num(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def make_key_from_values(img_id, hypo, gold):
    return (safe_text(img_id), safe_text(hypo), normalize_label(gold))


def make_key(row):
    return make_key_from_values(
        row.get("Flickr30K_ID", ""),
        row.get("hypothesis", ""),
        row.get("annotator_label", ""),
    )


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path, rows):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows, fieldnames=None):
    import csv
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def text_table(rows, columns):
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    body = [
        "  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns)
        for r in rows
    ]
    return "\n".join([header, sep] + body)


def scores_to_probs(scores):
    """Normalise a score dict to a distribution. Falls back to uniform."""
    if not isinstance(scores, dict):
        return {label: 1.0 / len(LABELS) for label in LABELS}

    values = []
    for label in LABELS:
        try:
            values.append(float(scores.get(label, 0.0)))
        except Exception:
            values.append(0.0)

    arr = np.asarray(values, dtype=float)
    if np.any(np.isnan(arr)) or arr.sum() <= 0:
        return {label: 1.0 / len(LABELS) for label in LABELS}

    arr = arr / arr.sum()
    return {LABELS[i]: float(arr[i]) for i in range(len(LABELS))}


def score_margin(scores):
    values = sorted([float(scores.get(label, 0.0)) for label in LABELS], reverse=True)
    if len(values) < 2:
        return 0.0
    return float(values[0] - values[1])


def score_argmax(scores):
    return max(LABELS, key=lambda label: float(scores.get(label, 0.0)))


def compute_metrics(golds, preds):
    """Accuracy, macro precision/recall/F1, and per-class recall."""
    if len(golds) != len(preds):
        raise ValueError("golds and preds must have the same length")

    n = len(golds)
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    accuracy = correct / n if n else 0.0

    per_class = {}
    precisions, recalls, f1s = [], [], []

    for label in LABELS:
        tp = sum(1 for g, p in zip(golds, preds) if g == label and p == label)
        fp = sum(1 for g, p in zip(golds, preds) if g != label and p == label)
        fn = sum(1 for g, p in zip(golds, preds) if g == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": tp + fn}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "n": n,
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
    }


def confusion_matrix_counts(golds, preds):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for g, p in zip(golds, preds):
        if g in LABEL_TO_ID and p in LABEL_TO_ID:
            matrix[LABEL_TO_ID[g], LABEL_TO_ID[p]] += 1
    return matrix


def row_normalize(matrix):
    matrix = matrix.astype(float)
    sums = matrix.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return matrix / sums