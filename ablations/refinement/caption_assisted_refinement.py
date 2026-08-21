import json
import os
import argparse
import re
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonlines
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.environ.get("DATA_ROOT", ".")
SPLIT = "dev"

DATASET_DIR = os.path.join(
    BASE_DIR,
    f"Output/{SPLIT}_dataset",
)

# Exact input files.
PREDICTION_FILE = os.path.join(
    DATASET_DIR,
    "qwen3_predictions_clean_v2",
    f"self_decompose_structured_{SPLIT}.jsonl",
)

DECOMPOSITION_FILE = os.path.join(
    DATASET_DIR,
    f"decompose_atoms_qwen32_{SPLIT}.jsonl",
)

CAPTION_FILE = os.path.join(
    BASE_DIR,
    "Output",
    "generated_captions_v2.jsonl",
)

# Output files.
OUTPUT_DIR = os.path.join(
    DATASET_DIR,
    "caption_assisted_refinement",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"caption_assisted_refinement_{SPLIT}.jsonl",
)

FAILED_FILE = os.path.join(
    OUTPUT_DIR,
    f"caption_assisted_refinement_{SPLIT}_failed.jsonl",
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    f"caption_assisted_refinement_{SPLIT}_summary.json",
)

# Same text-only model used for phrase extraction.
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
MAX_NEW_TOKENS = 160
MAX_ATTEMPTS = 2

# Set to 10 for an initial test, then None for the full dev set.
MAX_RECORDS = None

# True starts again from the beginning.
# False resumes from the existing output.
OVERWRITE = False

PROGRESS_EVERY = 50

SEP = "=" * 110


# ============================================================
# 2. BASIC HELPERS
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
        return json.dumps(value, ensure_ascii=False)

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
    ):
        return "neutral"

    return ""


def ensure_atomic_facts(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    facts: List[str] = []
    seen: Set[str] = set()

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

        if not fact:
            continue

        key = fact.lower()

        if key not in seen:
            seen.add(key)
            facts.append(fact)

    return facts


def format_atomic_facts(facts: List[str]) -> str:
    return "\n".join(
        f"A{index}. {fact}"
        for index, fact in enumerate(facts, start=1)
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


def validate_input_files() -> None:
    required_files = [
        PREDICTION_FILE,
        DECOMPOSITION_FILE,
        CAPTION_FILE,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )


# ============================================================
# 3. LOAD ATOMIC FACTS
# ============================================================

def load_decompositions() -> Dict[
    Tuple[str, str, str],
    List[str],
]:
    index: Dict[
        Tuple[str, str, str],
        List[str],
    ] = {}

    with jsonlines.open(
        DECOMPOSITION_FILE,
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

            atomic_facts = ensure_atomic_facts(
                record.get("atomic_facts", [])
            )

            if not image_id or not hypothesis:
                continue

            if not atomic_facts:
                continue

            index[
                make_key(
                    image_id,
                    hypothesis,
                    gold,
                )
            ] = atomic_facts

    return index


# ============================================================
# 4. LOAD DETAILED FLORENCE CAPTIONS
# ============================================================

def load_captions() -> Dict[str, str]:
    captions: Dict[str, str] = {}

    with jsonlines.open(
        CAPTION_FILE,
        "r",
    ) as reader:
        for record in reader:
            image_id = safe_text(
                record.get("Flickr30K_ID", "")
            )

            # The generated_captions_v2 file stores the detailed
            # caption directly in this field.
            caption = safe_text(
                record.get("generated_caption", "")
            )

            caption = re.sub(
                r"\s+",
                " ",
                caption,
            ).strip()

            if image_id and caption:
                captions[image_id] = caption

    return captions


# ============================================================
# 5. JUDGE OUTPUT PARSING
# ============================================================

def clean_model_output(text: Any) -> str:
    text = safe_text(text)

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = text.replace("```json", "```")
    text = text.replace("```JSON", "```")

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

    # Repair a possible trailing comma.
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


def parse_judge_output(
    raw_output: str,
    initial_label: str,
) -> Optional[Dict[str, str]]:
    parsed = extract_json_object(raw_output)

    if parsed is None:
        return None

    final_label = normalize_label(
        parsed.get("final_label", "")
    )

    if final_label not in LABELS:
        return None

    reason = safe_text(
        parsed.get("reason", "")
    )

    if not reason:
        reason = "No reason was provided by the judge."

    # Derive the decision mechanically.
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
# 6. JUDGE PROMPT
# ============================================================

def build_judge_prompt(
    hypothesis: str,
    atomic_facts: List[str],
    caption: str,
    initial_label: str,
    vlm_reasoning: str,
    retry: bool = False,
) -> str:
    retry_instruction = ""

    if retry:
        retry_instruction = (
            "\nThe previous response was invalid. "
            "Return only the required JSON object.\n"
        )

    return f"""You are a judge for a visual entailment task.

Input descriptions:

1. Full hypothesis:
   This is the complete claim that must be evaluated against the image.

2. Decomposed hypothesis claims:
   These are smaller parts of the full hypothesis.
   They help identify what must be checked.
   They are claims to evaluate, not evidence that the claims are true.

3. Detailed image caption:
   This is an automatically generated textual description of the original image.
   It provides supplementary visual evidence, but it may be incomplete.

4. Initial prediction:
   This is the visual-entailment label predicted by a vision-language model for
   the image and hypothesis.

5. VLM reasoning:
   This is the image-based reasoning provided by the vision-language model for
   its initial prediction.

Your task:
Judge whether the VLM's initial prediction should be kept or overturned.

Verify the full hypothesis and its decomposed claims against the supplementary
image caption. Also inspect the VLM reasoning to determine whether the VLM made
a reasoning or interpretation mistake.

Labels:
- entailment: the available visual evidence clearly supports all essential claims.
- neutral: at least one essential claim cannot be verified, and no essential claim
  is clearly contradicted.
- contradiction: at least one essential claim clearly conflicts with the available
  visual evidence.

Rules:
1. Treat the full hypothesis and decomposed claims only as claims to be evaluated.
2. Never use a decomposed claim as evidence that the claim is true.
3. Use the detailed caption as supplementary evidence about the image.
4. Use the VLM reasoning as the original model's interpretation of the image.
5. Check whether the VLM reasoning is consistent with the caption and the claims.
6. Keep the initial label when the supplied evidence supports the VLM's decision.
7. Overturn the initial label when the caption reveals a clear mistake, conflict,
   or unsupported conclusion in the VLM reasoning.
8. The caption may omit visible details. A detail missing from the caption is not
   automatically contradicted.
9. Use contradiction only when there is clear incompatible evidence.
10. Use neutral when the evidence is insufficient to verify or contradict an
    essential claim.
11. Do not use outside knowledge.
12. Do not invent image details.
13. Return only valid JSON.
{retry_instruction}
Full hypothesis:
{hypothesis}

Decomposed hypothesis claims:
{format_atomic_facts(atomic_facts)}

Supplementary detailed image caption:
{caption}

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
# 7. LOAD QWEN3-8B JUDGE
# ============================================================

def load_judge():
    print(SEP)
    print("LOADING QWEN3-8B CAPTION JUDGE")
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

    print("Judge loaded.")

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


def generate_judge_output(
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

    new_tokens = generated_ids[0][input_length:]

    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()


def judge_prediction(
    tokenizer,
    model,
    hypothesis: str,
    atomic_facts: List[str],
    caption: str,
    initial_label: str,
    vlm_reasoning: str,
) -> Dict[str, Any]:
    raw_outputs: List[str] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = build_judge_prompt(
            hypothesis=hypothesis,
            atomic_facts=atomic_facts,
            caption=caption,
            initial_label=initial_label,
            vlm_reasoning=vlm_reasoning,
            retry=(attempt > 1),
        )

        raw_output = generate_judge_output(
            tokenizer,
            model,
            prompt,
        )

        raw_outputs.append(raw_output)

        parsed = parse_judge_output(
            raw_output,
            initial_label,
        )

        if parsed is not None:
            return {
                **parsed,
                "parse_ok": True,
                "generation_attempt": attempt,
                "raw_model_output": raw_output,
            }

    # Conservative fallback if both outputs are invalid.
    return {
        "decision": "keep",
        "final_label": initial_label,
        "reason": (
            "The judge output could not be parsed, "
            "so the initial prediction was retained."
        ),
        "parse_ok": False,
        "generation_attempt": MAX_ATTEMPTS,
        "raw_model_output": (
            raw_outputs[-1]
            if raw_outputs
            else ""
        ),
    }


# ============================================================
# 8. RESUME LOGIC
# ============================================================

def load_processed_indices() -> Set[int]:
    processed: Set[int] = set()

    if not os.path.exists(OUTPUT_FILE):
        return processed

    with jsonlines.open(
        OUTPUT_FILE,
        "r",
    ) as reader:
        for record in reader.iter(
            type=dict,
            skip_invalid=True,
        ):
            source_index = record.get(
                "source_index"
            )

            if isinstance(source_index, int):
                processed.add(source_index)

    return processed


# ============================================================
# 9. EVALUATION
# ============================================================

def calculate_metrics() -> Dict[str, Any]:
    gold_labels: List[str] = []
    initial_labels: List[str] = []
    refined_labels: List[str] = []

    with jsonlines.open(
        OUTPUT_FILE,
        "r",
    ) as reader:
        for record in reader:
            gold = normalize_label(
                record.get("gold", "")
            )

            initial_label = normalize_label(
                record.get("initial_label", "")
            )

            refined_label = normalize_label(
                record.get("final_label", "")
            )

            if (
                gold not in LABELS
                or initial_label not in LABELS
                or refined_label not in LABELS
            ):
                continue

            gold_labels.append(gold)
            initial_labels.append(initial_label)
            refined_labels.append(refined_label)

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
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return summary


# ============================================================
# 10. MAIN LOOP
# ============================================================

def main() -> None:
    validate_input_files()

    if OVERWRITE:
        output_mode = "w"
        failed_mode = "w"
        processed_indices: Set[int] = set()
    else:
        output_mode = "a"
        failed_mode = "a"
        processed_indices = load_processed_indices()

    print(SEP)
    print("CAPTION-ASSISTED REFINEMENT")
    print(SEP)
    print(f"Predictions : {PREDICTION_FILE}")
    print(f"Atoms       : {DECOMPOSITION_FILE}")
    print(f"Captions    : {CAPTION_FILE}")
    print(f"Output      : {OUTPUT_FILE}")
    print(f"Already done: {len(processed_indices)}")
    print("")

    decomposition_index = load_decompositions()
    caption_index = load_captions()

    print(
        f"Loaded decompositions: "
        f"{len(decomposition_index):,}"
    )
    print(
        f"Loaded captions      : "
        f"{len(caption_index):,}"
    )
    print("")

    tokenizer, model = load_judge()

    seen = 0
    written = 0
    skipped = 0
    failed = 0

    with (
        jsonlines.open(
            PREDICTION_FILE,
            "r",
        ) as reader,

        jsonlines.open(
            OUTPUT_FILE,
            output_mode,
        ) as writer,

        jsonlines.open(
            FAILED_FILE,
            failed_mode,
        ) as failed_writer,
    ):
        for source_index, record in enumerate(reader):
            if (
                MAX_RECORDS is not None
                and written >= MAX_RECORDS
            ):
                break

            seen += 1

            if source_index in processed_indices:
                skipped += 1
                continue

            try:
                image_id = safe_text(
                    record.get("Flickr30K_ID", "")
                )

                hypothesis = safe_text(
                    record.get("hypothesis", "")
                )

                gold = normalize_label(
                    record.get("annotator_label", "")
                )

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

                decomposition_key = make_key(
                    image_id,
                    hypothesis,
                    gold,
                )

                atomic_facts = decomposition_index.get(
                    decomposition_key
                )

                if not atomic_facts:
                    raise ValueError(
                        "No matching atomic facts found."
                    )

                caption = caption_index.get(image_id)

                if not caption:
                    raise ValueError(
                        f"No caption found for image {image_id}."
                    )

                results = record.get(
                    "self_decompose_results",
                    {},
                )

                if not isinstance(results, dict):
                    raise ValueError(
                        "Missing self_decompose_results."
                    )

                initial_label = normalize_label(
                    results.get("score_prediction", "")
                )

                vlm_reasoning = safe_text(
                    results.get("reason", "")
                )

                if initial_label not in LABELS:
                    raise ValueError(
                        "Missing or invalid score_prediction."
                    )

                if not vlm_reasoning:
                    raise ValueError(
                        "Missing VLM reasoning."
                    )

                judgment = judge_prediction(
                    tokenizer=tokenizer,
                    model=model,
                    hypothesis=hypothesis,
                    atomic_facts=atomic_facts,
                    caption=caption,
                    initial_label=initial_label,
                    vlm_reasoning=vlm_reasoning,
                )

                output_record = {
                    "source_index": source_index,
                    "Flickr30K_ID": image_id,
                    "gold": gold,
                    "hypothesis": hypothesis,
                    "atomic_facts": atomic_facts,
                    "generated_caption": caption,

                    "initial_label": initial_label,
                    "vlm_reasoning": vlm_reasoning,

                    "decision": judgment["decision"],
                    "final_label": judgment["final_label"],
                    "judge_reason": judgment["reason"],

                    "parse_ok": judgment["parse_ok"],
                    "generation_attempt": judgment[
                        "generation_attempt"
                    ],
                    "raw_model_output": judgment[
                        "raw_model_output"
                    ],
                    "judge_model": JUDGE_MODEL_ID,
                }

                writer.write(output_record)
                writer._fp.flush()

                processed_indices.add(source_index)
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
                        "Flickr30K_ID": safe_text(
                            record.get(
                                "Flickr30K_ID",
                                "",
                            )
                        ),
                        "hypothesis": safe_text(
                            record.get(
                                "hypothesis",
                                "",
                            )
                        ),
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )

                failed_writer._fp.flush()

                print(
                    f"[FAILED] index={source_index} | "
                    f"ID={safe_text(record.get('Flickr30K_ID', ''))} | "
                    f"{error}"
                )

    summary = calculate_metrics()

    print("")
    print(SEP)
    print("CAPTION-ASSISTED REFINEMENT COMPLETE")
    print(SEP)
    print(f"Rows seen    : {seen}")
    print(f"Rows written : {written}")
    print(f"Rows skipped : {skipped}")
    print(f"Rows failed  : {failed}")
    print(f"Output       : {OUTPUT_FILE}")
    print(f"Summary      : {SUMMARY_FILE}")

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


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Caption-assisted refinement.")
    _ap.add_argument("--limit", type=int, default=None,
                     help="stop after N records, for testing")
    MAX_RECORDS = _ap.parse_args().limit
    main()