"""Decompose SNLI-VE hypotheses into atomic claims.

    python -m src.decomposition.decompose --model qwen32 --split train

Writes incrementally and resumes from the last completed record.
"""

import argparse
import json
import os
import re
import sys
import traceback

import jsonlines
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import config
from src.decomposition.prompts import SYSTEM_PROMPT_DECOMPOSE


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def count_existing_records(path):
    if not os.path.exists(path):
        return 0
    count = 0
    with jsonlines.open(path, "r") as reader:
        for _ in reader:
            count += 1
    return count


def extract_answer_bullets(text):
    if not text:
        return []
    try:
        block_match = re.search(r"<answer>([\s\S]*?)</answer>", text, flags=re.I)
        if not block_match:
            return []
        block = block_match.group(1).strip()
        atoms = []
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[-*\u2022]\s*", "", line).strip()
            if line:
                atoms.append(line)
        return atoms
    except Exception:
        return []


def extract_json_array(text):
    if not text:
        return []
    try:
        text = re.sub(r"```json|```", "", text).strip()
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, str)]
    except Exception:
        pass
    return []


def normalize_atom_list(atoms):
    if not isinstance(atoms, list):
        return []
    out, seen = [], set()
    for atom in atoms:
        if not isinstance(atom, str):
            continue
        a = re.sub(r"\s+", " ", atom.strip())
        if not a:
            continue
        if a[-1] not in ".!?":
            a += "."
        key = a.lower()
        if key not in seen:
            seen.add(key)
            out.append(a)
    return out


class AtomicRetriever:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find example bank: {path}")
        with open(path, "r", encoding="utf-8") as f:
            self.demons = json.load(f)
        self.sentences = list(self.demons.keys())
        self.bm25 = BM25Okapi([s.lower().split() for s in self.sentences])

    def get_top_k(self, query, k=4):
        top_sents = self.bm25.get_top_n(query.lower().split(), self.sentences, n=k)
        prompt = f"Reference Examples ({k} samples):\n\n"
        for s in top_sents:
            prompt += f"Claim: {s}\nAtomic Claims:\n"
            for atom in self.demons[s]:
                prompt += f"- {atom}\n"
            prompt += "\n"
        return prompt, top_sents


def load_model(model_key):
    spec = config.DECOMPOSER_MODELS[model_key]
    hf_id = spec["hf_id"]

    kwargs = {"trust_remote_code": True}
    if config.HF_TOKEN:
        kwargs["token"] = config.HF_TOKEN
    if config.HF_CACHE_DIR:
        kwargs["cache_dir"] = config.HF_CACHE_DIR

    if spec.get("gated") and not config.HF_TOKEN:
        raise RuntimeError(
            f"{hf_id} is gated. Set HF_TOKEN and accept the licence on the Hub."
        )

    print(f"Loading {hf_id} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        dtype=getattr(torch, spec["dtype"]),
        device_map="auto",
        **kwargs,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    return model, tokenizer, spec


def run_llm_decomposition(model, tokenizer, spec, few_shot_context, hypothesis):
    user_msg = f"{few_shot_context}\nClaim: {hypothesis}\nAtomic Claims:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_DECOMPOSE},
        {"role": "user", "content": user_msg},
    ]

    template_kwargs = {}
    if spec.get("disable_thinking"):
        template_kwargs["enable_thinking"] = False

    inputs_dict = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        max_length=config.MAX_INPUT_TOKENS,
        truncation=True,
        **template_kwargs,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs_dict,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs_dict["input_ids"].shape[-1]
    raw_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    atoms = extract_answer_bullets(raw_text)
    if not atoms:
        atoms = extract_json_array(raw_text)
    return normalize_atom_list(atoms), raw_text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="qwen32", choices=sorted(config.DECOMPOSER_MODELS))
    ap.add_argument("--split", default="train", choices=config.SPLITS)
    ap.add_argument("--limit", type=int, default=None, help="stop after N records")
    ap.add_argument("--input-file", default=None, help="override the input jsonl")
    ap.add_argument("--output-stem", default="decompose_atoms")
    args = ap.parse_args()

    input_file = args.input_file or config.input_file(args.split)
    success_file = config.output_file(args.output_stem, args.model, args.split)
    debug_file = config.output_file(args.output_stem, args.model, args.split, debug=True)
    progress_every = config.PROGRESS_EVERY[args.split]

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"No input at {input_file}. Is THESIS_ROOT set correctly? "
            f"(currently {config.THESIS_ROOT})"
        )

    ensure_parent_dir(success_file)
    ensure_parent_dir(debug_file)

    retriever = AtomicRetriever(config.DEMONS_JSON)
    model, tokenizer, spec = load_model(args.model)

    already_done = count_existing_records(success_file)
    print(f"Input:  {input_file}")
    print(f"Output: {success_file}")
    print(f"Resuming from record index: {already_done}", flush=True)

    with jsonlines.open(success_file, mode="a") as success_writer, \
         jsonlines.open(debug_file, mode="a") as debug_writer, \
         jsonlines.open(input_file) as reader:

        for i, obj in enumerate(reader):
            if args.limit is not None and i >= args.limit:
                break
            if i < already_done:
                continue

            flickr_id = obj.get("Flickr30K_ID")
            hypothesis = obj.get("sentence2", "")
            label = obj.get("gold_label") or obj.get("label")

            try:
                if not hypothesis or not isinstance(hypothesis, str):
                    raise ValueError("Missing or invalid hypothesis text")

                few_shot_context, retrieved = retriever.get_top_k(
                    hypothesis, k=config.TOP_K_EXAMPLES
                )
                atoms, raw_text = run_llm_decomposition(
                    model, tokenizer, spec, few_shot_context, hypothesis
                )

                if not atoms:
                    fallback = hypothesis.strip()
                    if fallback and fallback[-1] not in ".!?":
                        fallback += "."
                    atoms = [fallback] if fallback else []

                success_writer.write({
                    "Flickr30K_ID": flickr_id,
                    "annotator_label": label,
                    "hypothesis": hypothesis,
                    "atomic_facts": atoms,
                })
                success_writer._fp.flush()
                os.fsync(success_writer._fp.fileno())

                debug_writer.write({
                    "Flickr30K_ID": flickr_id,
                    "annotator_label": label,
                    "hypothesis": hypothesis,
                    "retrieved_example_hypotheses": retrieved,
                    "raw_model_output": raw_text,
                    "atomic_facts": atoms,
                })
                debug_writer._fp.flush()
                os.fsync(debug_writer._fp.fileno())

                if (i + 1) % progress_every == 0:
                    print(f"Progress: {i + 1} completed", flush=True)

            except Exception as e:
                print(f"ERROR at record {i} (Flickr30K_ID={flickr_id}): {e}", flush=True)
                debug_writer.write({
                    "Flickr30K_ID": flickr_id,
                    "annotator_label": label,
                    "hypothesis": hypothesis,
                    "error": str(e),
                    "traceback": traceback.format_exc()[:4000],
                })
                debug_writer._fp.flush()
                os.fsync(debug_writer._fp.fileno())
                continue

    print("Done.", flush=True)


if __name__ == "__main__":
    main()