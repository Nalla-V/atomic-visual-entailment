# internvl_qa_answer_generation.py

import argparse
import json
import os
import re
import traceback
from typing import Any, Dict, List, Set, Tuple

import jsonlines
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.environ.get("DATA_ROOT", ".")
IMAGE_DIR = os.path.join(BASE_DIR, "Input/flickr30k_images")

MODEL_ID = "OpenGVLab/InternVL3-8B-hf"

HF_CACHE_DIR = os.environ.get("HF_HOME") or None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

MAX_NEW_TOKENS = 250
PROGRESS_EVERY = 10

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

    input_jsonl = os.path.join(
        dataset_dir,
        "qa_assisted_ablation_v1",
        f"generated_questions_qwen3_8b_{split}.jsonl",
    )

    output_dir = os.path.join(
        dataset_dir,
        "qa_assisted_refinement",
    )
    os.makedirs(output_dir, exist_ok=True)

    output_jsonl = os.path.join(
        output_dir,
        f"qa_answers_internvl3_{split}.jsonl",
    )

    failed_jsonl = os.path.join(
        output_dir,
        f"qa_answers_internvl3_{split}_failed.jsonl",
    )

    return {
        "split": split,
        "input_jsonl": input_jsonl,
        "output_jsonl": output_jsonl,
        "failed_jsonl": failed_jsonl,
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
    text = safe_text(value).lower().strip()

    if text in {
        "entailment",
        "neutral",
        "contradiction",
    }:
        return text

    return text


def example_key_from_values(
    image_id: Any,
    hypothesis: Any,
    gold: Any,
) -> Tuple[str, str, str]:
    return (
        safe_text(image_id),
        safe_text(hypothesis),
        normalize_label(gold),
    )


def example_key(
    record: Dict[str, Any],
) -> Tuple[str, str, str]:
    return example_key_from_values(
        record.get("Flickr30K_ID", ""),
        record.get("hypothesis", ""),
        record.get(
            "gold",
            record.get("annotator_label", ""),
        ),
    )


def load_done_keys(
    output_path: str,
) -> Set[Tuple[str, str, str]]:
    done: Set[Tuple[str, str, str]] = set()

    if not os.path.exists(output_path):
        return done

    with jsonlines.open(output_path, "r") as reader:
        for record in reader.iter(
            type=dict,
            skip_invalid=True,
        ):
            done.add(example_key(record))

    return done


def load_image(
    image_id: str,
) -> Tuple[Image.Image, str]:
    image_path = os.path.join(
        IMAGE_DIR,
        f"{image_id}.jpg",
    )

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    return image, image_path


def get_questions(
    record: Dict[str, Any],
) -> List[str]:
    questions = record.get("questions", [])

    if not isinstance(questions, list):
        raise ValueError(
            "The questions field is not a list."
        )

    cleaned_questions = [
        safe_text(question)
        for question in questions
        if safe_text(question)
    ]

    if len(cleaned_questions) != 3:
        raise ValueError(
            "Each record must contain exactly three questions. "
            f"Found: {len(cleaned_questions)}"
        )

    return cleaned_questions


# ============================================================
# 4. INTERNVL PROMPT
# ============================================================

def build_qa_prompt(
    questions: List[str],
) -> str:
    return f"""Answer the following three questions using only what is visible in the image.

Rules:
1. Give one short and factual answer for each question.
2. Do not use outside knowledge.
3. Do not invent details that are not clearly visible.
4. If an answer cannot be determined from the image, answer: "Cannot be determined from the image."
5. Do not predict an entailment, neutral, or contradiction label.
6. Return only valid JSON.

Questions:
1. {questions[0]}
2. {questions[1]}
3. {questions[2]}

Return exactly:
{{
  "answers": [
    {{
      "question_id": 1,
      "answer": "..."
    }},
    {{
      "question_id": 2,
      "answer": "..."
    }},
    {{
      "question_id": 3,
      "answer": "..."
    }}
  ]
}}

Output:"""


# ============================================================
# 5. LOAD INTERNVL3
# ============================================================

def load_model():
    print(SEP)
    print("LOADING INTERNVL3 QA ANSWER MODEL")
    print(SEP)
    print(f"Model : {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"DTYPE : {DTYPE}")
    print("")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    print("InternVL3 model loaded.")

    return processor, model


# ============================================================
# 6. INTERNVL GENERATION
# ============================================================

def build_chat_prompt(
    processor,
    prompt_text: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt_text,
                },
            ],
        }
    ]

    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_answers(
    processor,
    model,
    image: Image.Image,
    questions: List[str],
) -> str:
    prompt_text = build_qa_prompt(questions)

    rendered_prompt = build_chat_prompt(
        processor=processor,
        prompt_text=prompt_text,
    )

    inputs = processor(
        text=rendered_prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_text = processor.batch_decode(
        generated_ids[:, input_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return generated_text


# ============================================================
# 7. OUTPUT PARSING
# ============================================================

def extract_json_object(
    response: str,
) -> Dict[str, Any]:
    response = safe_text(response)

    response = response.replace(
        "```json",
        "",
    ).replace(
        "```JSON",
        "",
    ).replace(
        "```",
        "",
    ).strip()

    match = re.search(
        r"\{.*\}",
        response,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(
            "No JSON object found in the model output."
        )

    parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError(
            "The model output is not a JSON object."
        )

    return parsed


def parse_answers(
    response: str,
    questions: List[str],
) -> List[Dict[str, Any]]:
    parsed = extract_json_object(response)

    raw_answers = parsed.get("answers")

    if not isinstance(raw_answers, list):
        raise ValueError(
            "The JSON output does not contain an answers list."
        )

    if len(raw_answers) != 3:
        raise ValueError(
            "The model must return exactly three answers. "
            f"Found: {len(raw_answers)}"
        )

    answer_by_id: Dict[int, str] = {}

    for item in raw_answers:
        if not isinstance(item, dict):
            raise ValueError(
                "Each answer must be a JSON object."
            )

        question_id = item.get("question_id")
        answer = safe_text(item.get("answer", ""))

        try:
            question_id = int(question_id)
        except Exception:
            raise ValueError(
                f"Invalid question_id: {question_id}"
            )

        if question_id not in {1, 2, 3}:
            raise ValueError(
                f"Unexpected question_id: {question_id}"
            )

        if not answer:
            raise ValueError(
                f"Empty answer for question {question_id}."
            )

        answer_by_id[question_id] = answer

    if set(answer_by_id.keys()) != {1, 2, 3}:
        raise ValueError(
            "The output must contain question IDs 1, 2, and 3."
        )

    return [
        {
            "question_id": question_id,
            "question": questions[question_id - 1],
            "answer": answer_by_id[question_id],
        }
        for question_id in [1, 2, 3]
    ]


# ============================================================
# 8. MAIN PROCESSING
# ============================================================

def run(split: str) -> None:
    paths = get_paths(split)

    input_path = paths["input_jsonl"]
    output_path = paths["output_jsonl"]
    failed_path = paths["failed_jsonl"]

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Generated-question file not found: {input_path}"
        )

    done_keys = load_done_keys(output_path)

    print(SEP)
    print(f"INTERNVL3 QA ANSWER GENERATION: {split.upper()}")
    print(SEP)
    print(f"Input       : {input_path}")
    print(f"Output      : {output_path}")
    print(f"Failed rows : {failed_path}")
    print(f"Already done: {len(done_keys)}")
    print("")

    processor, model = load_model()

    seen = 0
    written = 0
    skipped = 0
    failed = 0

    with (
        jsonlines.open(input_path, "r") as reader,
        jsonlines.open(output_path, "a") as writer,
        jsonlines.open(failed_path, "a") as failed_writer,
    ):
        for record in reader:
            seen += 1

            image_id = safe_text(
                record.get("Flickr30K_ID", "")
            )

            hypothesis = safe_text(
                record.get("hypothesis", "")
            )

            gold = normalize_label(
                record.get(
                    "gold",
                    record.get("annotator_label", ""),
                )
            )

            key = example_key_from_values(
                image_id,
                hypothesis,
                gold,
            )

            if key in done_keys:
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

                questions = get_questions(record)

                image, image_path = load_image(image_id)

                raw_output = generate_answers(
                    processor=processor,
                    model=model,
                    image=image,
                    questions=questions,
                )

                qa_pairs = parse_answers(
                    response=raw_output,
                    questions=questions,
                )

                output_record = {
                    "Flickr30K_ID": image_id,
                    "annotator_label": gold,
                    "gold": gold,
                    "hypothesis": hypothesis,
                    "atomic_facts": record.get(
                        "atomic_facts",
                        [],
                    ),
                    "questions": questions,
                    "qa_pairs": qa_pairs,
                    "num_questions": 3,
                    "num_answers": 3,
                    "image_path": image_path,
                    "answer_model": MODEL_ID,
                    "raw_model_output": raw_output,
                }

                writer.write(output_record)
                writer._fp.flush()

                done_keys.add(key)
                written += 1

                if written % PROGRESS_EVERY == 0:
                    print(
                        f"Written: {written} | "
                        f"Seen: {seen} | "
                        f"Skipped: {skipped} | "
                        f"Failed: {failed}"
                    )

                image.close()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as error:
                failed += 1

                failed_record = {
                    "Flickr30K_ID": image_id,
                    "annotator_label": gold,
                    "hypothesis": hypothesis,
                    "questions": record.get(
                        "questions",
                        [],
                    ),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }

                failed_writer.write(failed_record)
                failed_writer._fp.flush()

                print(
                    f"[FAILED] image={image_id} | "
                    f"error={error}"
                )

    print("")
    print(SEP)
    print("INTERNVL3 QA ANSWER GENERATION COMPLETE")
    print(SEP)
    print(f"Rows seen    : {seen}")
    print(f"Rows written : {written}")
    print(f"Rows skipped : {skipped}")
    print(f"Rows failed  : {failed}")
    print(f"Output       : {output_path}")
    print(f"Failed rows  : {failed_path}")


# ============================================================
# 9. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use InternVL3-8B to answer the three generated "
            "visual questions for each image."
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