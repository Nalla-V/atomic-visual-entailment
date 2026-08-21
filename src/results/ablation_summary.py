# evaluate_refinement_methods.py

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple

import jsonlines
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.environ.get("DATA_ROOT", ".")

LABELS = [
    "entailment",
    "neutral",
    "contradiction",
]

SEP = "=" * 100


# ============================================================
# 2. PATHS
# ============================================================

def get_paths(split: str) -> Dict[str, str]:
    dataset_dir = os.path.join(
        BASE_DIR,
        f"Output/{split}_dataset",
    )

    qa_file = os.path.join(
        dataset_dir,
        "qa_assisted_refinement",
        f"qa_assisted_refinement_{split}.jsonl",
    )

    caption_file = os.path.join(
        dataset_dir,
        "caption_assisted_refinement",
        f"caption_assisted_refinement_{split}.jsonl",
    )

    output_dir = os.path.join(
        dataset_dir,
        "ablation_refinement_summary",
    )
    os.makedirs(output_dir, exist_ok=True)

    return {
        "qa_file": qa_file,
        "caption_file": caption_file,

        "summary_json": os.path.join(
            output_dir,
            f"refinement_metrics_{split}.json",
        ),

        "summary_csv": os.path.join(
            output_dir,
            f"refinement_metrics_{split}.csv",
        ),
    }


# ============================================================
# 3. HELPERS
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def normalize_label(value: Any) -> str:
    text = safe_text(value).lower()

    if text in LABELS:
        return text

    if (
        "entailment" in text
        or "entailed" in text
        or "supported" in text
    ):
        return "entailment"

    if (
        "contradiction" in text
        or "contradict" in text
        or "incompatible" in text
        or "conflict" in text
    ):
        return "contradiction"

    if (
        "neutral" in text
        or "uncertain" in text
        or "insufficient" in text
        or "unsupported" in text
    ):
        return "neutral"

    return ""


def validate_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )


def record_key(
    record: Dict[str, Any],
) -> Tuple[Any, ...]:
    """
    source_index should uniquely identify the original dev row.

    A composite key is used only if source_index is unavailable.
    """

    source_index = record.get("source_index")

    if source_index is not None:
        return (
            "source_index",
            int(source_index),
        )

    return (
        "composite",
        safe_text(record.get("Flickr30K_ID", "")),
        safe_text(record.get("hypothesis", "")),
        normalize_label(
            record.get(
                "gold",
                record.get("annotator_label", ""),
            )
        ),
        int(record.get("row_key_occurrence", 0)),
    )


# ============================================================
# 4. LOAD REFINEMENT OUTPUT
# ============================================================

def load_refinement_file(
    path: str,
    method_name: str,
) -> Tuple[
    Dict[Tuple[Any, ...], Dict[str, Any]],
    int,
]:
    records: Dict[
        Tuple[Any, ...],
        Dict[str, Any],
    ] = {}

    invalid_rows = 0

    with jsonlines.open(path, "r") as reader:
        for record in reader.iter(
            type=dict,
            skip_invalid=True,
        ):
            gold = normalize_label(
                record.get(
                    "gold",
                    record.get("annotator_label", ""),
                )
            )

            initial_label = normalize_label(
                record.get("initial_label", "")
            )

            final_label = normalize_label(
                record.get("final_label", "")
            )

            if (
                gold not in LABELS
                or initial_label not in LABELS
                or final_label not in LABELS
            ):
                invalid_rows += 1
                continue

            key = record_key(record)

            if key in records:
                raise ValueError(
                    f"Duplicate key found in {method_name}: {key}"
                )

            records[key] = {
                "source_index": record.get("source_index"),
                "Flickr30K_ID": safe_text(
                    record.get("Flickr30K_ID", "")
                ),
                "hypothesis": safe_text(
                    record.get("hypothesis", "")
                ),
                "gold": gold,
                "initial_label": initial_label,
                "final_label": final_label,
                "decision": safe_text(
                    record.get("decision", "")
                ).lower(),
            }

    return records, invalid_rows


# ============================================================
# 5. METRIC CALCULATION
# ============================================================

def calculate_prediction_metrics(
    gold_labels: List[str],
    predictions: List[str],
) -> Dict[str, float]:
    if not gold_labels:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
        }

    return {
        "accuracy": float(
            accuracy_score(
                gold_labels,
                predictions,
            )
        ),

        "macro_f1": float(
            f1_score(
                gold_labels,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }


def calculate_method_summary(
    records: Dict[
        Tuple[Any, ...],
        Dict[str, Any],
    ],
    invalid_rows: int,
) -> Dict[str, Any]:
    rows = list(records.values())

    gold_labels = [
        row["gold"]
        for row in rows
    ]

    initial_labels = [
        row["initial_label"]
        for row in rows
    ]

    final_labels = [
        row["final_label"]
        for row in rows
    ]

    initial_metrics = calculate_prediction_metrics(
        gold_labels,
        initial_labels,
    )

    refined_metrics = calculate_prediction_metrics(
        gold_labels,
        final_labels,
    )

    return {
        "num_evaluated": len(rows),
        "num_invalid_rows": invalid_rows,

        "initial": initial_metrics,
        "refined": refined_metrics,

        "change": {
            "accuracy": (
                refined_metrics["accuracy"]
                - initial_metrics["accuracy"]
            ),
            "macro_f1": (
                refined_metrics["macro_f1"]
                - initial_metrics["macro_f1"]
            ),
        },
    }


# ============================================================
# 6. COMMON-ROW COMPARISON
# ============================================================

def calculate_common_summary(
    qa_records: Dict[
        Tuple[Any, ...],
        Dict[str, Any],
    ],
    caption_records: Dict[
        Tuple[Any, ...],
        Dict[str, Any],
    ],
) -> Dict[str, Any]:
    common_keys = sorted(
        set(qa_records.keys())
        & set(caption_records.keys()),
        key=str,
    )

    valid_common_keys = []
    identity_mismatches = 0
    initial_label_mismatches = 0

    for key in common_keys:
        qa_row = qa_records[key]
        caption_row = caption_records[key]

        same_record = (
            qa_row["Flickr30K_ID"]
            == caption_row["Flickr30K_ID"]
            and qa_row["hypothesis"]
            == caption_row["hypothesis"]
            and qa_row["gold"]
            == caption_row["gold"]
        )

        if not same_record:
            identity_mismatches += 1
            continue

        if (
            qa_row["initial_label"]
            != caption_row["initial_label"]
        ):
            initial_label_mismatches += 1

        valid_common_keys.append(key)

    gold_labels = [
        qa_records[key]["gold"]
        for key in valid_common_keys
    ]

    qa_initial_labels = [
        qa_records[key]["initial_label"]
        for key in valid_common_keys
    ]

    qa_final_labels = [
        qa_records[key]["final_label"]
        for key in valid_common_keys
    ]

    caption_initial_labels = [
        caption_records[key]["initial_label"]
        for key in valid_common_keys
    ]

    caption_final_labels = [
        caption_records[key]["final_label"]
        for key in valid_common_keys
    ]

    qa_initial_metrics = calculate_prediction_metrics(
        gold_labels,
        qa_initial_labels,
    )

    qa_refined_metrics = calculate_prediction_metrics(
        gold_labels,
        qa_final_labels,
    )

    caption_initial_metrics = calculate_prediction_metrics(
        gold_labels,
        caption_initial_labels,
    )

    caption_refined_metrics = calculate_prediction_metrics(
        gold_labels,
        caption_final_labels,
    )

    return {
        "num_common_rows": len(valid_common_keys),

        "identity_mismatches_excluded": (
            identity_mismatches
        ),

        "initial_label_mismatches": (
            initial_label_mismatches
        ),

        "qa_assisted": {
            "initial": qa_initial_metrics,
            "refined": qa_refined_metrics,

            "change": {
                "accuracy": (
                    qa_refined_metrics["accuracy"]
                    - qa_initial_metrics["accuracy"]
                ),
                "macro_f1": (
                    qa_refined_metrics["macro_f1"]
                    - qa_initial_metrics["macro_f1"]
                ),
            },
        },

        "caption_assisted": {
            "initial": caption_initial_metrics,
            "refined": caption_refined_metrics,

            "change": {
                "accuracy": (
                    caption_refined_metrics["accuracy"]
                    - caption_initial_metrics["accuracy"]
                ),
                "macro_f1": (
                    caption_refined_metrics["macro_f1"]
                    - caption_initial_metrics["macro_f1"]
                ),
            },
        },
    }


# ============================================================
# 7. SAVE CSV
# ============================================================

def save_csv(
    summary: Dict[str, Any],
    output_path: str,
) -> None:
    rows = []

    for method_name in [
        "qa_assisted",
        "caption_assisted",
    ]:
        method_summary = summary[
            "all_available_rows"
        ][method_name]

        for prediction_stage in [
            "initial",
            "refined",
        ]:
            metrics = method_summary[
                prediction_stage
            ]

            rows.append(
                {
                    "scope": "all_available_rows",
                    "method": method_name,
                    "prediction_stage": prediction_stage,
                    "num_examples": method_summary[
                        "num_evaluated"
                    ],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )

    common_summary = summary["common_rows"]

    for method_name in [
        "qa_assisted",
        "caption_assisted",
    ]:
        for prediction_stage in [
            "initial",
            "refined",
        ]:
            metrics = common_summary[
                method_name
            ][prediction_stage]

            rows.append(
                {
                    "scope": "common_rows",
                    "method": method_name,
                    "prediction_stage": prediction_stage,
                    "num_examples": common_summary[
                        "num_common_rows"
                    ],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scope",
                "method",
                "prediction_stage",
                "num_examples",
                "accuracy",
                "macro_f1",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_change(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def print_method_results(
    title: str,
    summary: Dict[str, Any],
) -> None:
    print(title)
    print("-" * 70)

    print(
        f"Evaluated rows     : "
        f"{summary['num_evaluated']:,}"
    )

    print(
        f"Initial accuracy   : "
        f"{format_percentage(summary['initial']['accuracy'])}"
    )

    print(
        f"Initial macro-F1   : "
        f"{format_percentage(summary['initial']['macro_f1'])}"
    )

    print(
        f"Refined accuracy   : "
        f"{format_percentage(summary['refined']['accuracy'])}"
    )

    print(
        f"Refined macro-F1   : "
        f"{format_percentage(summary['refined']['macro_f1'])}"
    )

    print(
        f"Accuracy change    : "
        f"{format_change(summary['change']['accuracy'])}"
    )

    print(
        f"Macro-F1 change    : "
        f"{format_change(summary['change']['macro_f1'])}"
    )

    print("")


# ============================================================
# 9. MAIN
# ============================================================

def run(split: str) -> None:
    paths = get_paths(split)

    validate_file(paths["qa_file"])
    validate_file(paths["caption_file"])

    print(SEP)
    print(
        f"REFINEMENT METRICS SUMMARY: "
        f"{split.upper()}"
    )
    print(SEP)
    print(f"QA file      : {paths['qa_file']}")
    print(f"Caption file : {paths['caption_file']}")
    print("")

    qa_records, qa_invalid = load_refinement_file(
        paths["qa_file"],
        method_name="QA-assisted refinement",
    )

    caption_records, caption_invalid = (
        load_refinement_file(
            paths["caption_file"],
            method_name="Caption-assisted refinement",
        )
    )

    qa_summary = calculate_method_summary(
        qa_records,
        qa_invalid,
    )

    caption_summary = calculate_method_summary(
        caption_records,
        caption_invalid,
    )

    common_summary = calculate_common_summary(
        qa_records,
        caption_records,
    )

    summary = {
        "split": split,

        "files": {
            "qa_assisted": paths["qa_file"],
            "caption_assisted": paths[
                "caption_file"
            ],
        },

        "all_available_rows": {
            "qa_assisted": qa_summary,
            "caption_assisted": caption_summary,
        },

        "common_rows": common_summary,
    }

    with open(
        paths["summary_json"],
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    save_csv(
        summary,
        paths["summary_csv"],
    )

    print_method_results(
        "QA-ASSISTED REFINEMENT",
        qa_summary,
    )

    print_method_results(
        "CAPTION-ASSISTED REFINEMENT",
        caption_summary,
    )

    print("COMMON-ROW COMPARISON")
    print("-" * 70)
    print(
        f"Common evaluated rows       : "
        f"{common_summary['num_common_rows']:,}"
    )
    print(
        f"Identity mismatches excluded: "
        f"{common_summary['identity_mismatches_excluded']}"
    )
    print(
        f"Initial-label mismatches    : "
        f"{common_summary['initial_label_mismatches']}"
    )
    print("")

    print_method_results(
        "QA-ASSISTED — COMMON ROWS",
        {
            "num_evaluated": common_summary[
                "num_common_rows"
            ],
            **common_summary["qa_assisted"],
        },
    )

    print_method_results(
        "CAPTION-ASSISTED — COMMON ROWS",
        {
            "num_evaluated": common_summary[
                "num_common_rows"
            ],
            **common_summary["caption_assisted"],
        },
    )

    print(SEP)
    print("SUMMARY SAVED")
    print(SEP)
    print(f"JSON: {paths['summary_json']}")
    print(f"CSV : {paths['summary_csv']}")


# ============================================================
# 10. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate accuracy and macro-F1 for "
            "caption-assisted and QA-assisted refinement."
        )
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Dataset split. Default: dev",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.split)