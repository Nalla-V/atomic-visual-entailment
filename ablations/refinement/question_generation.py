# generate_qa_questions_qwen3_v1.py

import argparse
import hashlib
import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.environ.get("DATA_ROOT", ".")

HF_CACHE_DIR = os.environ.get("HF_HOME") or None

# Same text-only model used by the grounding phrase extractor.
QUESTION_MODEL_ID = "Qwen/Qwen3-8B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

NUM_QUESTIONS = 3

MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 180

# One normal attempt and one stricter retry.
MAX_GENERATION_ATTEMPTS = 2

PROGRESS_EVERY = 50

SEP = "=" * 120


# ============================================================
# 2. SPLIT PATH CONFIGURATION
# ============================================================

def get_split_config(split: str) -> Dict[str, str]:
    split = split.lower().strip()

    if split not in {"dev", "test"}:
        raise ValueError("split must be either 'dev' or 'test'")

    dataset_dir = os.path.join(
        BASE_DIR,
        f"Output/{split}_dataset",
    )

    output_dir = os.path.join(
        dataset_dir,
        "qa_assisted_ablation_v1",
    )
    os.makedirs(output_dir, exist_ok=True)

    return {
        "split": split,
        "dataset_dir": dataset_dir,

        "input_jsonl": os.path.join(
            dataset_dir,
            f"decompose_atoms_qwen32_{split}.jsonl",
        ),

        "output_jsonl": os.path.join(
            output_dir,
            f"generated_questions_qwen3_8b_{split}.jsonl",
        ),

        "failed_jsonl": os.path.join(
            output_dir,
            f"generated_questions_qwen3_8b_{split}_failed.jsonl",
        ),
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
        return json.dumps(value, ensure_ascii=False)

    return str(value).strip()


def normalize_label(value: Any) -> str:
    """
    Used only to preserve the source gold label in the output.

    The gold label is never passed to the question-generation model.
    """
    text = safe_text(value).lower().strip()

    if text in {"entailment", "neutral", "contradiction"}:
        return text

    if (
        "entailment" in text
        or "entailed" in text
        or "support" in text
    ):
        return "entailment"

    if (
        "contradiction" in text
        or "contradict" in text
        or "conflict" in text
    ):
        return "contradiction"

    if (
        "neutral" in text
        or "uncertain" in text
        or "insufficient" in text
    ):
        return "neutral"

    return text


def ensure_list_of_atoms(atoms: Any) -> List[str]:
    """
    Accept atomic facts stored as strings or dictionaries.
    """
    if not isinstance(atoms, list):
        return []

    output: List[str] = []
    seen: Set[str] = set()

    for atom in atoms:
        if isinstance(atom, str):
            text = atom.strip()

        elif isinstance(atom, dict):
            text = safe_text(
                atom.get(
                    "atom_text",
                    atom.get(
                        "atom",
                        atom.get(
                            "text",
                            atom.get(
                                "claim",
                                atom.get("fact", ""),
                            ),
                        ),
                    ),
                )
            )

        else:
            text = ""

        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            continue

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        output.append(text)

    return output


def format_atomic_facts(atoms: List[str]) -> str:
    return "\n".join(
        f"A{index}. {atom}"
        for index, atom in enumerate(atoms, start=1)
    )


def make_record_key(
    image_id: str,
    hypothesis: str,
    gold: str,
    atomic_facts: List[str],
) -> str:
    """
    Stable key used for safe resume.

    Atomic facts are included so that a record is regenerated if
    its decomposition changes.
    """
    payload = {
        "Flickr30K_ID": image_id,
        "hypothesis": hypothesis,
        "gold": gold,
        "atomic_facts": atomic_facts,
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha1(encoded).hexdigest()


def count_jsonl_rows(path: str) -> int:
    count = 0

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1

    return count


# ============================================================
# 4. MODEL OUTPUT CLEANING
# ============================================================

def clean_model_output(text: Any) -> str:
    text = safe_text(text)

    # Remove Qwen thinking blocks if one appears unexpectedly.
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


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = clean_model_output(text)

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        return None

    raw = match.group(0)

    try:
        parsed = json.loads(raw)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    # Basic repair for trailing commas.
    try:
        repaired = re.sub(
            r",\s*([}\]])",
            r"\1",
            raw,
        )

        parsed = json.loads(repaired)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        return None

    return None


def extract_json_array(text: str) -> Optional[List[Any]]:
    """
    Fallback for outputs such as:
    ["Question 1?", "Question 2?", "Question 3?"]
    """
    text = clean_model_output(text)

    match = re.search(
        r"\[.*\]",
        text,
        flags=re.DOTALL,
    )

    if not match:
        return None

    raw = match.group(0)

    try:
        parsed = json.loads(raw)

        if isinstance(parsed, list):
            return parsed

    except Exception:
        pass

    try:
        repaired = re.sub(
            r",\s*([\]])",
            r"\1",
            raw,
        )

        parsed = json.loads(repaired)

        if isinstance(parsed, list):
            return parsed

    except Exception:
        return None

    return None


def clean_question(question: Any) -> str:
    """
    Normalize one generated question.
    """
    text = safe_text(question)

    if not text:
        return ""

    text = text.strip().strip('"').strip("'")

    # Remove list numbering and question identifiers.
    text = re.sub(
        r"^\s*(?:[-*•]\s*)?",
        "",
        text,
    )

    text = re.sub(
        r"^\s*(?:Q(?:uestion)?\s*)?\d+\s*[\).:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not text:
        return ""

    # Remove an accidental answer if the model writes:
    # "Is a dog visible? Answer: Yes"
    if "?" in text:
        text = text[: text.find("?") + 1]
    else:
        text = text.rstrip(".!;:") + "?"

    return text.strip()


def deduplicate_questions(questions: List[Any]) -> List[str]:
    output: List[str] = []
    seen: Set[str] = set()

    for question in questions:
        cleaned = clean_question(question)

        if not cleaned:
            continue

        key = re.sub(
            r"[^a-z0-9]+",
            " ",
            cleaned.lower(),
        ).strip()

        if not key or key in seen:
            continue

        seen.add(key)
        output.append(cleaned)

    return output


def extract_questions_from_text(text: str) -> List[str]:
    """
    Extract questions from JSON first, then use text lines as a fallback.
    """
    text = clean_model_output(text)

    candidate_questions: List[Any] = []

    parsed_object = extract_json_object(text)

    if parsed_object is not None:
        raw_questions = parsed_object.get(
            "questions",
            parsed_object.get(
                "visual_questions",
                parsed_object.get("generated_questions", []),
            ),
        )

        if isinstance(raw_questions, list):
            candidate_questions = raw_questions

        elif isinstance(raw_questions, dict):
            candidate_questions = list(raw_questions.values())

    if not candidate_questions:
        parsed_array = extract_json_array(text)

        if parsed_array is not None:
            candidate_questions = parsed_array

    if not candidate_questions:
        # Final fallback for numbered or bulleted text.
        for line in text.splitlines():
            line = line.strip()

            if not line or "?" not in line:
                continue

            candidate_questions.append(line)

    questions = deduplicate_questions(candidate_questions)

    # If more than three were produced, retain only the first three.
    if len(questions) >= NUM_QUESTIONS:
        return questions[:NUM_QUESTIONS]

    return questions


def validate_questions(
    questions: List[str],
) -> Tuple[bool, str]:
    if len(questions) != NUM_QUESTIONS:
        return (
            False,
            f"Expected exactly {NUM_QUESTIONS} questions, "
            f"but extracted {len(questions)}.",
        )

    if len(set(question.lower() for question in questions)) != NUM_QUESTIONS:
        return False, "Generated questions are not distinct."

    for question in questions:
        if not question.endswith("?"):
            return False, f"Question does not end with '?': {question}"

        if len(question.split()) < 3:
            return False, f"Question is too short: {question}"

    return True, ""


# ============================================================
# 5. QUESTION-GENERATION PROMPT
# ============================================================

def build_question_prompt(
    hypothesis: str,
    atomic_facts: List[str],
    retry: bool = False,
) -> str:
    atoms_text = format_atomic_facts(atomic_facts)

    retry_instruction = ""

    if retry:
        retry_instruction = (
            "\nImportant: the previous formatting was invalid. "
            "Return exactly one JSON object containing exactly "
            "three question strings.\n"
        )

    return f"""You generate visual questions for a visual entailment task.

Given the complete hypothesis and its atomic facts, generate exactly three short questions whose answers from the image would provide the most useful evidence for deciding whether the complete hypothesis is entailment, neutral, or contradiction.

Label interpretation:
- Entailment: the image clearly supports all essential claims.
- Contradiction: the image clearly conflicts with at least one essential claim.
- Neutral: the image does not provide enough evidence for at least one essential claim and does not clearly conflict with it.

Rules:
1. Generate exactly three questions.
2. The questions must be answerable by looking at the image.
3. The three questions should collectively cover the most important content in the full hypothesis and atomic facts.
4. Ask about concrete visible entities, counts, attributes, actions, relationships, or setting.
5. Do not assume that the hypothesis is true.
6. Avoid duplicate or strongly overlapping questions.
7. Do not answer the questions.
8. Return only valid JSON in the required format.
{retry_instruction}
Required format:
{{
  "questions": [
    "Question 1?",
    "Question 2?",
    "Question 3?"
  ]
}}

Full hypothesis:
{hypothesis}

Atomic facts:
{atoms_text}

Output:"""


# ============================================================
# 6. MODEL LOADING
# ============================================================

def load_question_model():
    print(SEP)
    print("LOADING QWEN3 QUESTION-GENERATION MODEL")
    print(SEP)
    print(f"Model : {QUESTION_MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"DTYPE : {DTYPE}")
    print("")

    tokenizer = AutoTokenizer.from_pretrained(
        QUESTION_MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        QUESTION_MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    print("Question-generation model loaded.")

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


# ============================================================
# 7. QUESTION GENERATION
# ============================================================

def run_single_generation(
    tokenizer,
    model,
    prompt: str,
) -> str:
    rendered_prompt = render_chat_prompt(
        tokenizer=tokenizer,
        prompt=prompt,
    )

    inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    model_device = next(model.parameters()).device

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

    generated_tokens = generated_ids[0][input_length:]

    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return raw_output


def generate_three_questions(
    tokenizer,
    model,
    hypothesis: str,
    atomic_facts: List[str],
) -> Dict[str, Any]:
    attempt_outputs: List[Dict[str, Any]] = []

    for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
        prompt = build_question_prompt(
            hypothesis=hypothesis,
            atomic_facts=atomic_facts,
            retry=(attempt > 1),
        )

        raw_output = run_single_generation(
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
        )

        questions = extract_questions_from_text(raw_output)

        valid, validation_error = validate_questions(questions)

        attempt_outputs.append(
            {
                "attempt": attempt,
                "raw_output": raw_output,
                "extracted_questions": questions,
                "valid": valid,
                "validation_error": validation_error,
            }
        )

        if valid:
            return {
                "questions": questions,
                "generation_attempt": attempt,
                "raw_model_output": raw_output,
                "attempt_outputs": attempt_outputs,
            }

    errors = [
        attempt.get("validation_error", "")
        for attempt in attempt_outputs
    ]

    raise ValueError(
        "Could not generate exactly three valid questions after "
        f"{MAX_GENERATION_ATTEMPTS} attempts. Errors: {errors}"
    )


# ============================================================
# 8. RESUME HELPERS
# ============================================================

def load_processed_keys(output_path: str) -> Set[str]:
    processed: Set[str] = set()

    if not os.path.exists(output_path):
        return processed

    try:
        with jsonlines.open(output_path, mode="r") as reader:
            for record in reader.iter(
                type=dict,
                skip_invalid=True,
            ):
                key = safe_text(record.get("record_key", ""))

                if key:
                    processed.add(key)
                    continue

                # Backward-compatible fallback.
                image_id = safe_text(
                    record.get("Flickr30K_ID", "")
                )

                hypothesis = safe_text(
                    record.get("hypothesis", "")
                )

                gold = normalize_label(
                    record.get(
                        "annotator_label",
                        record.get("gold", ""),
                    )
                )

                atomic_facts = ensure_list_of_atoms(
                    record.get("atomic_facts", [])
                )

                if image_id and hypothesis:
                    processed.add(
                        make_record_key(
                            image_id=image_id,
                            hypothesis=hypothesis,
                            gold=gold,
                            atomic_facts=atomic_facts,
                        )
                    )

    except Exception as error:
        print(
            "[WARNING] Existing output could not be read fully: "
            f"{error}"
        )

    return processed


# ============================================================
# 9. MAIN PROCESSING
# ============================================================

def run(args):
    config = get_split_config(args.split)

    input_path = config["input_jsonl"]
    output_path = config["output_jsonl"]
    failed_path = config["failed_jsonl"]

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input decomposition file not found: {input_path}"
        )

    if args.overwrite:
        processed_keys: Set[str] = set()
        output_mode = "w"
        failed_mode = "w"
    else:
        processed_keys = load_processed_keys(output_path)
        output_mode = "a"
        failed_mode = "a"

    tokenizer, model = load_question_model()

    total_input_rows = count_jsonl_rows(input_path)

    print("")
    print(SEP)
    print(
        f"GENERATING THREE QUESTIONS FOR "
        f"{args.split.upper()}"
    )
    print(SEP)
    print(f"Input JSONL     : {input_path}")
    print(f"Output JSONL    : {output_path}")
    print(f"Failed JSONL    : {failed_path}")
    print(f"Input rows      : {total_input_rows}")
    print(f"Already done    : {len(processed_keys)}")
    print(f"Overwrite       : {args.overwrite}")
    print(f"Maximum new rows: {args.max_records}")
    print("")

    seen = 0
    skipped = 0
    written = 0
    failed = 0

    with (
        jsonlines.open(input_path, mode="r") as reader,
        jsonlines.open(output_path, mode=output_mode) as writer,
        jsonlines.open(failed_path, mode=failed_mode) as failed_writer,
    ):
        for source_index, record in enumerate(reader):
            if (
                args.max_records is not None
                and written >= args.max_records
            ):
                break

            seen += 1

            try:
                image_id = safe_text(
                    record.get("Flickr30K_ID", "")
                )

                hypothesis = safe_text(
                    record.get(
                        "hypothesis",
                        record.get("sentence2", ""),
                    )
                )

                gold = normalize_label(
                    record.get(
                        "annotator_label",
                        record.get("gold", ""),
                    )
                )

                atomic_facts = ensure_list_of_atoms(
                    record.get(
                        "atomic_facts",
                        record.get("raw_atoms", []),
                    )
                )

                if not image_id:
                    raise ValueError(
                        "The record has no Flickr30K_ID."
                    )

                if not hypothesis:
                    raise ValueError(
                        "The record has no hypothesis."
                    )

                atom_source = "decomposition_file"

                if not atomic_facts:
                    # A simple hypothesis may occasionally have no
                    # parsed atom list. Preserve coverage by using H.
                    atomic_facts = [hypothesis]
                    atom_source = "hypothesis_fallback"

                record_key = make_record_key(
                    image_id=image_id,
                    hypothesis=hypothesis,
                    gold=gold,
                    atomic_facts=atomic_facts,
                )

                if record_key in processed_keys:
                    skipped += 1
                    continue

                generated = generate_three_questions(
                    tokenizer=tokenizer,
                    model=model,
                    hypothesis=hypothesis,
                    atomic_facts=atomic_facts,
                )

                questions = generated["questions"]

                # Final safeguard before writing.
                valid, validation_error = validate_questions(
                    questions
                )

                if not valid:
                    raise ValueError(validation_error)

                output_record = {
                    "source_index": source_index,
                    "record_key": record_key,

                    "row_id": record.get("row_id", None),
                    "Flickr30K_ID": image_id,

                    "annotator_label": record.get(
                        "annotator_label",
                        gold,
                    ),
                    "gold": gold,

                    "hypothesis": hypothesis,
                    "atomic_facts": atomic_facts,
                    "atomic_facts_source": atom_source,

                    "questions": questions,
                    "num_questions": len(questions),

                    "question_generation": {
                        "model": QUESTION_MODEL_ID,
                        "generation_attempt": generated[
                            "generation_attempt"
                        ],
                        "max_input_tokens": MAX_INPUT_TOKENS,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "do_sample": False,
                        "thinking_enabled": False,
                        "prompt_version": "qa_question_generation_v1",
                    },

                    "raw_model_output": generated[
                        "raw_model_output"
                    ],

                    "generation_attempt_outputs": generated[
                        "attempt_outputs"
                    ],
                }

                writer.write(output_record)
                writer._fp.flush()

                processed_keys.add(record_key)
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

                failed_record = {
                    "source_index": source_index,
                    "row_id": record.get("row_id", None),

                    "Flickr30K_ID": safe_text(
                        record.get("Flickr30K_ID", "")
                    ),

                    "annotator_label": safe_text(
                        record.get(
                            "annotator_label",
                            record.get("gold", ""),
                        )
                    ),

                    "hypothesis": safe_text(
                        record.get(
                            "hypothesis",
                            record.get("sentence2", ""),
                        )
                    ),

                    "atomic_facts": ensure_list_of_atoms(
                        record.get(
                            "atomic_facts",
                            record.get("raw_atoms", []),
                        )
                    ),

                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }

                failed_writer.write(failed_record)
                failed_writer._fp.flush()

                print(
                    f"[FAILED] source_index={source_index} | "
                    f"ID={failed_record['Flickr30K_ID']} | "
                    f"{type(error).__name__}: {error}"
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("")
    print(SEP)
    print("QUESTION GENERATION COMPLETE")
    print(SEP)
    print(f"Input records seen : {seen}")
    print(f"Records written    : {written}")
    print(f"Records skipped    : {skipped}")
    print(f"Records failed     : {failed}")
    print(f"Output JSONL       : {output_path}")
    print(f"Failed JSONL       : {failed_path}")


# ============================================================
# 10. COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate exactly three visual questions from "
            "each full hypothesis and its atomic facts."
        )
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test"],
        default="dev",
        help="Dataset split to process. Default: dev",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help=(
            "Maximum number of new records to generate. "
            "Use a small value for testing."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite the existing output and failed files. "
            "Without this flag, the script resumes."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())