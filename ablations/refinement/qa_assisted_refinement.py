# qa_assisted_refinement.py

import argparse
import json
import os
import re
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonlines
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.environ.get("DATA_ROOT", ".")

JUDGE_MODEL_ID = "Qwen/Qwen3-8B"

HF_CACHE_DIR = os.environ.get("HF_HOME") or None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

LABELS = [
    "entailment",
    "neutral",
    "contradiction",
]

MAX_INPUT_TOKENS = 3072
MAX_NEW_TOKENS = 180

PROGRESS_EVERY = 50

SEP = "=" * 120


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

def get_paths(split: str) -> Dict[str, str]:
    split = split.lower().strip()

    if split not in {"dev", "test"}:
        raise ValueError("split must be either 'dev' or 'test'")

    dataset_dir = os.path.join(
        BASE_DIR,
        f"Output/{split}_dataset",
    )

    prediction_file = os.path.join(
        dataset_dir,
        "qwen3_predictions_clean_v2",
        f"self_decompose_structured_{split}.jsonl",
    )

    qa_file = os.path.join(
        dataset_dir,
        "qa_assisted_refinement",
        f"qa_answers_internvl3_{split}.jsonl",
    )

    output_dir = os.path.join(
        dataset_dir,
        "qa_assisted_refinement",
    )
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"qa_assisted_refinement_{split}.jsonl",
    )

    failed_file = os.path.join(
        output_dir,
        f"qa_assisted_refinement_{split}_failed.jsonl",
    )

    summary_file = os.path.join(
        output_dir,
        f"qa_assisted_refinement_{split}_summary.json",
    )

    return {
        "split": split,
        "prediction_file": prediction_file,
        "qa_file": qa_file,
        "output_file": output_file,
        "failed_file": failed_file,
        "summary_file": summary_file,
    }


# ============================================================
# 3. BASIC HELPERS
# ============================================================

def safe_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(
            str(item).strip()
            for item in value
            if str(item).strip()
        )

    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value).strip()


def normalize_label(value: Any) -> str:
    text = safe_text(value).lower().strip()

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
        or "conflict" in text
        or "incompatible" in text
    ):
        return "contradiction"

    if (
        "neutral" in text
        or "uncertain" in text
        or "insufficient" in text
        or "unsupported" in text
        or "unclear" in text
    ):
        return "neutral"

    return ""


def ensure_atomic_facts(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    facts: List[str] = []

    for item in value:
        if isinstance(item, str):
            fact = item.strip()

        elif isinstance(item, dict):
            fact = safe_text(
                item.get(
                    "atom_text",
                    item.get(
                        "atom",
                        item.get(
                            "text",
                            item.get(
                                "claim",
                                item.get("fact", ""),
                            ),
                        ),
                    ),
                )
            )

        else:
            fact = ""

        fact = re.sub(r"\s+", " ", fact).strip()

        if fact:
            facts.append(fact)

    return facts


def format_atomic_facts(
    atomic_facts: List[str],
) -> str:
    return "\n".join(
        f"A{index}. {fact}"
        for index, fact in enumerate(
            atomic_facts,
            start=1,
        )
    )


def make_key(
    image_id: Any,
    hypothesis: Any,
    gold: Any,
) -> Tuple[str, str, str]:
    return (
        safe_text(image_id),
        safe_text(hypothesis),
        normalize_label(gold),
    )


def make_output_key(
    image_id: Any,
    hypothesis: Any,
    gold: Any,
    occurrence: Any,
) -> Tuple[str, str, str, int]:
    return (
        safe_text(image_id),
        safe_text(hypothesis),
        normalize_label(gold),
        int(occurrence),
    )


def validate_files(paths: Dict[str, str]) -> None:
    required_files = [
        paths["prediction_file"],
        paths["qa_file"],
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )


# ============================================================
# 4. LOAD INITIAL PREDICTIONS
# ============================================================

def load_prediction_index(
    prediction_file: str,
) -> Dict[
    Tuple[str, str, str],
    List[Dict[str, str]],
]:
    """
    A list is stored for every key so duplicate dataset rows
    are not lost.
    """

    index: Dict[
        Tuple[str, str, str],
        List[Dict[str, str]],
    ] = defaultdict(list)

    with jsonlines.open(
        prediction_file,
        "r",
    ) as reader:
        for record in reader:
            image_id = safe_text(
                record.get("Flickr30K_ID", "")
            )

            hypothesis = safe_text(
                record.get("hypothesis", "")
            )

            gold = normalize_label(
                record.get("annotator_label", "")
            )

            results = record.get(
                "self_decompose_results",
                {},
            )

            if not isinstance(results, dict):
                continue

            initial_label = normalize_label(
                results.get("score_prediction", "")
            )

            vlm_reasoning = safe_text(
                results.get("reason", "")
            )

            if (
                not image_id
                or not hypothesis
                or gold not in LABELS
                or initial_label not in LABELS
            ):
                continue

            key = make_key(
                image_id,
                hypothesis,
                gold,
            )

            index[key].append(
                {
                    "initial_label": initial_label,
                    "vlm_reasoning": vlm_reasoning,
                }
            )

    return index


# ============================================================
# 5. QA PAIR EXTRACTION
# ============================================================

def get_qa_pairs(
    record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    raw_pairs = record.get("qa_pairs", [])

    if not isinstance(raw_pairs, list):
        raise ValueError(
            "qa_pairs is not a list."
        )

    if len(raw_pairs) != 3:
        raise ValueError(
            "Expected exactly three QA pairs. "
            f"Found: {len(raw_pairs)}"
        )

    pairs: List[Dict[str, Any]] = []

    for index, item in enumerate(
        raw_pairs,
        start=1,
    ):
        if not isinstance(item, dict):
            raise ValueError(
                "Each QA pair must be a JSON object."
            )

        question_id = item.get(
            "question_id",
            index,
        )

        try:
            question_id = int(question_id)
        except Exception:
            raise ValueError(
                f"Invalid question_id: {question_id}"
            )

        question = safe_text(
            item.get("question", "")
        )

        answer = safe_text(
            item.get("answer", "")
        )

        if not question:
            raise ValueError(
                f"Missing question for QA pair {index}."
            )

        if not answer:
            raise ValueError(
                f"Missing answer for QA pair {index}."
            )

        pairs.append(
            {
                "question_id": question_id,
                "question": question,
                "answer": answer,
            }
        )

    pairs.sort(
        key=lambda item: item["question_id"]
    )

    expected_ids = [1, 2, 3]

    actual_ids = [
        item["question_id"]
        for item in pairs
    ]

    if actual_ids != expected_ids:
        raise ValueError(
            "QA pairs must contain question IDs 1, 2, and 3."
        )

    return pairs


def format_qa_pairs(
    qa_pairs: List[Dict[str, Any]],
) -> str:
    sections: List[str] = []

    for pair in qa_pairs:
        question_id = pair["question_id"]
        question = pair["question"]
        answer = pair["answer"]

        sections.append(
            f"Q{question_id}: {question}\n"
            f"A{question_id}: {answer}"
        )

    return "\n\n".join(sections)


# ============================================================
# 6. JUDGE PROMPT
# ============================================================

def build_judge_prompt(
    hypothesis: str,
    atomic_facts: List[str],
    qa_pairs: List[Dict[str, Any]],
    initial_label: str,
    vlm_reasoning: str,
) -> str:
    return f"""You are a judge for a visual entailment task.

Input descriptions:

1. Full hypothesis:
   This is the complete claim that must be evaluated against the image.

2. Decomposed hypothesis claims:
   These are smaller parts of the full hypothesis.
   They identify what must be checked.
   They are claims to evaluate and are not evidence that the claims are true.

3. Supplementary question-answer evidence:
   The questions were generated from the hypothesis and decomposed claims.
   The answers were produced by an independent vision-language model after
   examining the original image.
   The questions are checks and are not evidence by themselves.
   The answers provide supplementary evidence about the image.

4. Initial prediction:
   This is the visual-entailment label predicted by the original
   vision-language model for the image and hypothesis.

5. Initial VLM reasoning:
   This is the image-based reasoning supplied by the original
   vision-language model for its initial prediction.

Your task:
Judge whether the initial prediction should be kept or overturned.

Verify the hypothesis and its decomposed claims using the supplementary
QA answers. Inspect the initial VLM reasoning and determine whether the
original model made a visual or reasoning mistake.

Labels:
- entailment: the visual evidence clearly supports all essential claims.
- neutral: at least one essential claim cannot be verified, and no
  essential claim is clearly contradicted.
- contradiction: at least one essential claim clearly conflicts with
  the visual evidence.

Rules:
1. Treat the hypothesis and decomposed claims only as claims to evaluate.
2. Use the QA answers as supplementary evidence from the image.
3. Use the initial VLM reasoning as the original model's interpretation
   of the image, but check whether that reasoning contains a mistake.
4. Keep the initial prediction when the initial reasoning and QA evidence
   support the same decision.
5. Overturn the initial prediction when the QA evidence reveals a clear
   visual or reasoning error in the original decision.
6. A QA answer may be incomplete. Missing information is not automatically
   a contradiction.
7. Use contradiction only when there is clear incompatible evidence.
8. Use neutral when the available evidence is insufficient.
9. Do not use outside knowledge.
10. Do not invent image details.
11. Return only valid JSON.

Full hypothesis:
{hypothesis}

Decomposed hypothesis claims:
{format_atomic_facts(atomic_facts)}

Supplementary QA evidence:
{format_qa_pairs(qa_pairs)}

Initial VLM prediction:
{initial_label}

Initial VLM reasoning:
{vlm_reasoning}

Determine whether the initial prediction should be kept or overturned.

Return exactly:
{{
  "final_label": "entailment" or "neutral" or "contradiction",
  "reason": "brief explanation of your judgement"
}}

Output:"""


# ============================================================
# 7. MODEL OUTPUT PARSING
# ============================================================

def clean_model_output(
    text: Any,
) -> str:
    text = safe_text(text)

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = text.replace(
        "```json",
        "```",
    ).replace(
        "```JSON",
        "```",
    )

    text = re.sub(
        r"```(.*?)```",
        r"\1",
        text,
        flags=re.DOTALL,
    )

    return text.strip()


def extract_json_object(
    text: str,
) -> Optional[Dict[str, Any]]:
    text = clean_model_output(text)

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        return None

    raw_json = match.group(0)

    try:
        parsed = json.loads(raw_json)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    try:
        repaired = re.sub(
            r",\s*([}\]])",
            r"\1",
            raw_json,
        )

        parsed = json.loads(repaired)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        return None

    return None


def parse_judgment(
    raw_output: str,
    initial_label: str,
) -> Dict[str, str]:
    parsed = extract_json_object(raw_output)

    if parsed is None:
        raise ValueError(
            "No valid JSON object found in judge output."
        )

    final_label = normalize_label(
        parsed.get("final_label", "")
    )

    if final_label not in LABELS:
        raise ValueError(
            "Judge returned an invalid final_label."
        )

    reason = safe_text(
        parsed.get("reason", "")
    )

    if not reason:
        raise ValueError(
            "Judge returned an empty reason."
        )

    decision = (
        "keep"
        if final_label == initial_label
        else "overturn"
    )

    return {
        "decision": decision,
        "final_label": final_label,
        "reason": reason,
    }


# ============================================================
# 8. LOAD QWEN3-8B JUDGE
# ============================================================

def load_judge():
    print(SEP)
    print("LOADING QWEN3-8B QA JUDGE")
    print(SEP)
    print(f"Model : {JUDGE_MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"DTYPE : {DTYPE}")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(
        JUDGE_MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    print("Judge model loaded.")

    return tokenizer, model


def render_chat_prompt(
    tokenizer,
    prompt: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_judgment(
    tokenizer,
    model,
    prompt: str,
) -> str:
    rendered_prompt = render_chat_prompt(
        tokenizer,
        prompt,
    )

    inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    model_device = next(
        model.parameters()
    ).device

    inputs = {
        key: value.to(model_device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = generated_ids[0][
        input_length:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


# ============================================================
# 9. RESUME LOGIC
# ============================================================

def load_completed_rows(
    output_file: str,
) -> Set[Tuple[str, str, str, int]]:
    completed: Set[
        Tuple[str, str, str, int]
    ] = set()

    if not os.path.exists(output_file):
        return completed

    with jsonlines.open(
        output_file,
        "r",
    ) as reader:
        for record in reader.iter(
            type=dict,
            skip_invalid=True,
        ):
            occurrence = record.get(
                "row_key_occurrence",
                0,
            )

            completed.add(
                make_output_key(
                    record.get("Flickr30K_ID", ""),
                    record.get("hypothesis", ""),
                    record.get(
                        "gold",
                        record.get(
                            "annotator_label",
                            "",
                        ),
                    ),
                    occurrence,
                )
            )

    return completed


# ============================================================
# 10. EVALUATION
# ============================================================

def calculate_metrics(
    output_file: str,
    summary_file: str,
) -> Dict[str, Any]:
    gold_labels: List[str] = []
    initial_labels: List[str] = []
    refined_labels: List[str] = []

    if not os.path.exists(output_file):
        return {}

    with jsonlines.open(
        output_file,
        "r",
    ) as reader:
        for record in reader.iter(
            type=dict,
            skip_invalid=True,
        ):
            gold = normalize_label(
                record.get("gold", "")
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
                continue

            gold_labels.append(gold)
            initial_labels.append(initial_label)
            refined_labels.append(final_label)

    if not gold_labels:
        return {}

    summary = {
        "num_examples": len(gold_labels),

        "initial_accuracy": float(
            accuracy_score(
                gold_labels,
                initial_labels,
            )
        ),

        "initial_macro_f1": float(
            f1_score(
                gold_labels,
                initial_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),

        "refined_accuracy": float(
            accuracy_score(
                gold_labels,
                refined_labels,
            )
        ),

        "refined_macro_f1": float(
            f1_score(
                gold_labels,
                refined_labels,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )
        ),
    }

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return summary


# ============================================================
# 11. MAIN PROCESSING
# ============================================================

def run(split: str) -> None:
    paths = get_paths(split)
    validate_files(paths)

    prediction_index = load_prediction_index(
        paths["prediction_file"]
    )

    completed_rows = load_completed_rows(
        paths["output_file"]
    )

    print(SEP)
    print(f"QA-ASSISTED REFINEMENT: {split.upper()}")
    print(SEP)
    print(f"Initial predictions: {paths['prediction_file']}")
    print(f"QA evidence        : {paths['qa_file']}")
    print(f"Output             : {paths['output_file']}")
    print(f"Failed rows        : {paths['failed_file']}")
    print(f"Already completed  : {len(completed_rows)}")
    print("")

    tokenizer, model = load_judge()

    occurrence_counts: Dict[
        Tuple[str, str, str],
        int,
    ] = defaultdict(int)

    seen = 0
    written = 0
    skipped = 0
    failed = 0

    with (
        jsonlines.open(
            paths["qa_file"],
            "r",
        ) as qa_reader,

        jsonlines.open(
            paths["output_file"],
            "a",
        ) as writer,

        jsonlines.open(
            paths["failed_file"],
            "a",
        ) as failed_writer,
    ):
        for source_index, qa_record in enumerate(
            qa_reader
        ):
            seen += 1

            image_id = safe_text(
                qa_record.get("Flickr30K_ID", "")
            )

            hypothesis = safe_text(
                qa_record.get("hypothesis", "")
            )

            gold = normalize_label(
                qa_record.get(
                    "gold",
                    qa_record.get(
                        "annotator_label",
                        "",
                    ),
                )
            )

            base_key = make_key(
                image_id,
                hypothesis,
                gold,
            )

            occurrence = occurrence_counts[
                base_key
            ]

            occurrence_counts[
                base_key
            ] += 1

            output_key = make_output_key(
                image_id,
                hypothesis,
                gold,
                occurrence,
            )

            if output_key in completed_rows:
                skipped += 1
                continue

            try:
                if not image_id:
                    raise ValueError(
                        "Missing Flickr30K_ID."
                    )

                if not hypothesis:
                    raise ValueError(
                        "Missing hypothesis."
                    )

                if gold not in LABELS:
                    raise ValueError(
                        f"Invalid gold label: {gold}"
                    )

                prediction_rows = prediction_index.get(
                    base_key,
                    [],
                )

                if occurrence >= len(prediction_rows):
                    raise ValueError(
                        "No matching initial prediction found "
                        f"for occurrence {occurrence}."
                    )

                initial_record = prediction_rows[
                    occurrence
                ]

                initial_label = initial_record[
                    "initial_label"
                ]

                vlm_reasoning = initial_record[
                    "vlm_reasoning"
                ]

                if not vlm_reasoning:
                    raise ValueError(
                        "Initial VLM reasoning is empty."
                    )

                atomic_facts = ensure_atomic_facts(
                    qa_record.get(
                        "atomic_facts",
                        [],
                    )
                )

                if not atomic_facts:
                    raise ValueError(
                        "Atomic facts are missing."
                    )

                qa_pairs = get_qa_pairs(
                    qa_record
                )

                prompt = build_judge_prompt(
                    hypothesis=hypothesis,
                    atomic_facts=atomic_facts,
                    qa_pairs=qa_pairs,
                    initial_label=initial_label,
                    vlm_reasoning=vlm_reasoning,
                )

                raw_output = generate_judgment(
                    tokenizer=tokenizer,
                    model=model,
                    prompt=prompt,
                )

                judgment = parse_judgment(
                    raw_output=raw_output,
                    initial_label=initial_label,
                )

                output_record = {
                    "source_index": source_index,
                    "row_key_occurrence": occurrence,

                    "Flickr30K_ID": image_id,
                    "annotator_label": gold,
                    "gold": gold,

                    "hypothesis": hypothesis,
                    "atomic_facts": atomic_facts,
                    "qa_pairs": qa_pairs,

                    "initial_label": initial_label,
                    "vlm_reasoning": vlm_reasoning,

                    "decision": judgment["decision"],
                    "final_label": judgment[
                        "final_label"
                    ],
                    "judge_reason": judgment["reason"],

                    "judge_model": JUDGE_MODEL_ID,
                    "raw_model_output": raw_output,
                }

                writer.write(output_record)
                writer._fp.flush()

                completed_rows.add(output_key)
                written += 1

                if written % PROGRESS_EVERY == 0:
                    print(
                        f"Written: {written} | "
                        f"Seen: {seen} | "
                        f"Skipped: {skipped} | "
                        f"Failed: {failed}"
                    )

            except Exception as error:
                failed += 1

                failed_writer.write(
                    {
                        "source_index": source_index,
                        "row_key_occurrence": occurrence,
                        "Flickr30K_ID": image_id,
                        "annotator_label": gold,
                        "hypothesis": hypothesis,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )

                failed_writer._fp.flush()

                print(
                    f"[FAILED] index={source_index} | "
                    f"image={image_id} | "
                    f"error={error}"
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = calculate_metrics(
        output_file=paths["output_file"],
        summary_file=paths["summary_file"],
    )

    print("")
    print(SEP)
    print("QA-ASSISTED REFINEMENT COMPLETE")
    print(SEP)
    print(f"Rows seen    : {seen}")
    print(f"Rows written : {written}")
    print(f"Rows skipped : {skipped}")
    print(f"Rows failed  : {failed}")
    print(f"Output       : {paths['output_file']}")
    print(f"Failed rows  : {paths['failed_file']}")
    print(f"Summary      : {paths['summary_file']}")

    if summary:
        print("")
        print(
            f"Initial accuracy : "
            f"{summary['initial_accuracy'] * 100:.2f}%"
        )
        print(
            f"Initial macro-F1 : "
            f"{summary['initial_macro_f1'] * 100:.2f}%"
        )
        print(
            f"Refined accuracy : "
            f"{summary['refined_accuracy'] * 100:.2f}%"
        )
        print(
            f"Refined macro-F1 : "
            f"{summary['refined_macro_f1'] * 100:.2f}%"
        )


# ============================================================
# 12. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use Qwen3-8B to judge whether the initial "
            "Qwen self-decompose prediction should be "
            "kept or overturned using InternVL QA evidence."
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