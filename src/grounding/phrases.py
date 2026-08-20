"""Extract grounding phrases from the selected candidate's reasoning.

    python -m src.grounding.phrases --split dev

Qwen3-8B turns each evidence item into a short visual phrase for Grounding
DINO, plus a shorter match phrase used later for Flickr30k Entities matching.
Stops once each groundable label reaches its limit = 2500.
"""

import argparse
import os
import sys
import traceback
from collections import Counter

import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import config
from src.grounding import common as C
from src.grounding.prompts import build_phrase_prompt

PHRASE_MODEL_ID = "Qwen/Qwen3-8B"

MAX_ATOM_ITEMS_PER_ROW = 4
TARGET_PER_LABEL = 2500
REQUIRE_GROUNDABLE_PHRASE = False
PROGRESS_EVERY = 10
MAX_NEW_TOKENS = 80
MAX_INPUT_TOKENS = 2048

SEP = "=" * 120
DASH = "-" * 120


def build_paths(split, eval_name):
    if split not in {"dev", "test"}:
        raise ValueError("split must be either 'dev' or 'test'")

    dataset_dir = os.path.join(config.OUTPUT_DIR, f"{split}_dataset")
    eval_dir = os.path.join(dataset_dir, eval_name)
    grounding_dir = os.path.join(eval_dir, "grounding_dino_v1")
    os.makedirs(grounding_dir, exist_ok=True)

    return {
        "split": split,
        "eval_dir": eval_dir,
        "grounding_dir": grounding_dir,
        "input_jsonl": os.path.join(eval_dir, f"ave_ls_v3_{split}_grounding_input.jsonl"),
        "output_jsonl": os.path.join(
            grounding_dir, f"grounding_phrase_ave_ls_final_{split}.jsonl"),
        "failed_jsonl": os.path.join(
            grounding_dir, f"grounding_phrase_ave_ls_final_{split}_failed_rows.jsonl"),
        "skipped_jsonl": os.path.join(
            grounding_dir, f"grounding_phrase_ave_ls_final_{split}_skipped_rows.jsonl"),
    }


# ============================================================
# EVIDENCE ITEMS
# ============================================================

def normalize_evidence_item(item, fallback_label, fallback_reason,
                            fallback_atom, index):
    atom = C.safe_text(item.get("atom", item.get("selected_atom",
                       item.get("claim", item.get("fact", fallback_atom)))))
    atom_label = C.normalize_label(item.get("atom_label",
                                   item.get("selected_atom_label",
                                   item.get("label",
                                   item.get("prediction", fallback_label)))))
    vlm_reasoning = C.safe_text(item.get("vlm_reasoning",
                                item.get("selected_atom_reason",
                                item.get("reason", fallback_reason))))
    evidence_source = C.safe_text(item.get("evidence_source",
                                           "selected_output_evidence"))

    return {
        "evidence_index": int(item.get("evidence_index", index)),
        "atom": atom if atom else fallback_atom,
        "atom_label": atom_label if atom_label in C.LABELS else fallback_label,
        "vlm_reasoning": vlm_reasoning,
        "evidence_source": evidence_source,
    }


def get_evidence_items_from_record(rec):
    """Accepts both a flattened record and one nesting the items under
    grounding_input."""
    final_label = C.normalize_label(rec.get("final_label", rec.get("prediction", "")))
    full_reason = C.safe_text(rec.get("reason", ""))

    atoms = C.ensure_list_of_atoms(rec.get("atomic_facts", []))
    fallback_atom = atoms[0] if atoms else C.safe_text(rec.get("hypothesis", ""))

    raw_items = rec.get("evidence_items", None)
    if not isinstance(raw_items, list):
        grounding_input = rec.get("grounding_input", {})
        raw_items = (grounding_input.get("evidence_items", [])
                     if isinstance(grounding_input, dict) else [])
    if not isinstance(raw_items, list):
        raw_items = []

    out = []
    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        norm = normalize_evidence_item(item, final_label, full_reason,
                                       fallback_atom, idx)
        if norm["atom"] or norm["vlm_reasoning"]:
            out.append(norm)

    return out[:MAX_ATOM_ITEMS_PER_ROW]


# ============================================================
# MODEL
# ============================================================

def load_phrase_model():
    print(SEP)
    print("LOADING GROUNDING PHRASE EXTRACTION MODEL")
    print(SEP)
    print(f"Model : {PHRASE_MODEL_ID}")

    kwargs = {"trust_remote_code": True}
    if config.HF_TOKEN:
        kwargs["token"] = config.HF_TOKEN
    if config.HF_CACHE_DIR:
        kwargs["cache_dir"] = config.HF_CACHE_DIR

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Dtype : {dtype}\n")

    tokenizer = AutoTokenizer.from_pretrained(PHRASE_MODEL_ID, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        PHRASE_MODEL_ID, dtype=dtype, device_map="auto", **kwargs).eval()

    print("Phrase extraction model loaded.\n")
    return tokenizer, model


def render_chat_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    try:
        # Qwen3 emits <think> blocks unless thinking mode is switched off.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate_grounding_phrase(tokenizer, model, final_label, hypothesis, atom,
                              vlm_reasoning, full_reason, selected_model,
                              selected_method):
    prompt = build_phrase_prompt(final_label, hypothesis, atom, vlm_reasoning,
                                 full_reason, selected_model, selected_method)
    rendered = render_chat_prompt(tokenizer, prompt)

    inputs = tokenizer(rendered, return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_TOKENS)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                   do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)

    input_len = inputs["input_ids"].shape[1]
    raw_output = tokenizer.decode(generated[0][input_len:],
                                  skip_special_tokens=True).strip()

    parsed = C.extract_json_object(raw_output)

    if parsed is not None:
        phrase = C.clean_phrase(parsed.get("phrase", ""))
        match_phrase = C.clean_match_phrase(parsed.get("match_phrase", ""))
        can_ground = C.bool_field(parsed.get("can_ground", phrase.upper() != "NONE"),
                                  default=phrase.upper() != "NONE")
        source = "llm_json"
    else:
        phrase = C.clean_phrase(raw_output)
        match_phrase = C.fallback_match_phrase(phrase)
        can_ground = phrase.upper() != "NONE"
        source = "llm_text"

    if C.phrase_is_bad(phrase):
        phrase = C.fallback_phrase(atom, f"{vlm_reasoning} {full_reason}", final_label)
        match_phrase = C.fallback_match_phrase(phrase)
        can_ground = phrase.upper() != "NONE"
        source = "fallback"

    if phrase.upper() == "NONE":
        phrase, match_phrase, can_ground = "NONE", "NONE", False

    if match_phrase.upper() == "NONE" and can_ground:
        match_phrase = C.fallback_match_phrase(phrase)

    return {
        "can_ground": can_ground,
        "phrase": phrase,
        "match_phrase": match_phrase,
        "phrase_source": source,
        "raw_model_output": raw_output,
        "phrase_model": PHRASE_MODEL_ID,
    }


# ============================================================
# OUTPUT HELPERS
# ============================================================

def build_legacy_grounding_phrase(grounding_phrases):
    """One flat grounding_phrase object, kept so the detection stage can read a
    single phrase per row. Picks the first groundable phrase, else the first."""
    if not grounding_phrases:
        return {
            "final_prediction": "", "score_prediction": "", "selected_atom": "",
            "selected_atom_label": "", "selected_atom_reason": "",
            "can_ground": False, "phrase": "NONE", "match_phrase": "NONE",
            "phrase_source": "", "raw_model_output": "",
            "phrase_model": PHRASE_MODEL_ID,
        }

    chosen = None
    for item in grounding_phrases:
        if (item.get("can_ground", False)
                and C.safe_text(item.get("phrase", "")).upper() != "NONE"):
            chosen = item
            break
    if chosen is None:
        chosen = grounding_phrases[0]

    return {
        "final_prediction": chosen.get("final_prediction", chosen.get("final_label", "")),
        "score_prediction": chosen.get("score_prediction", ""),
        "selected_atom": chosen.get("atom", ""),
        "selected_atom_label": chosen.get("atom_label", ""),
        "selected_atom_reason": chosen.get("vlm_reasoning", ""),
        "can_ground": chosen.get("can_ground", False),
        "phrase": chosen.get("phrase", "NONE"),
        "match_phrase": chosen.get("match_phrase", chosen.get("phrase", "NONE")),
        "phrase_source": chosen.get("phrase_source", ""),
        "raw_model_output": chosen.get("raw_model_output", ""),
        "phrase_model": chosen.get("phrase_model", PHRASE_MODEL_ID),
    }


def grounding_phrase_summary(grounding_phrases):
    groundable = [i for i in grounding_phrases
                  if i.get("can_ground", False)
                  and C.safe_text(i.get("phrase", "")).upper() != "NONE"]
    with_match = [i for i in groundable
                  if C.safe_text(i.get("match_phrase", "")).upper() != "NONE"]
    return {
        "num_phrase_items": len(grounding_phrases),
        "num_groundable_phrase_items": len(groundable),
        "num_items_with_match_phrase": len(with_match),
        "max_atom_items_per_row": MAX_ATOM_ITEMS_PER_ROW,
        "phrase_model": PHRASE_MODEL_ID,
    }


def write_skip(skipped_writer, rec, reason):
    skipped_writer.write({
        "row_id": rec.get("row_id", None),
        "Flickr30K_ID": C.safe_text(rec.get("Flickr30K_ID", "")),
        "hypothesis": C.safe_text(rec.get("hypothesis", "")),
        "gold": C.normalize_label(rec.get("gold", rec.get("annotator_label", ""))),
        "final_label": C.normalize_label(rec.get("final_label", rec.get("prediction", ""))),
        "selected_candidate": C.safe_text(rec.get("selected_candidate", "")),
        "candidate_matches_learned_label": rec.get("candidate_matches_learned_label", None),
        "grounding_eligible": rec.get("grounding_eligible", None),
        "skip_reason": reason,
    })
    skipped_writer._fp.flush()


def should_stop(counts, target):
    return all(counts[label] >= target for label in C.GROUNDABLE_FINAL_LABELS)


# ============================================================
# MAIN
# ============================================================

def run(split, eval_name, target_per_label, limit):
    paths = build_paths(split, eval_name)

    print(SEP)
    print(f"GROUNDING PHRASE EXTRACTION: {split.upper()}")
    print(SEP)
    print(f"Input : {paths['input_jsonl']}")
    print(f"Output: {paths['output_jsonl']}")
    print(f"Target per label: {target_per_label}\n")

    if not os.path.exists(paths["input_jsonl"]):
        raise FileNotFoundError(
            f"No grounding input at {paths['input_jsonl']}. "
            f"Run src.selection.evaluate first.")

    tokenizer, model = load_phrase_model()

    counts = Counter({lab: 0 for lab in C.GROUNDABLE_FINAL_LABELS})
    skip_reasons = Counter()
    processed = skipped = failed = 0
    total_phrase_items = total_groundable = 0

    with jsonlines.open(paths["output_jsonl"], "w") as out_writer, \
         jsonlines.open(paths["failed_jsonl"], "w") as failed_writer, \
         jsonlines.open(paths["skipped_jsonl"], "w") as skipped_writer, \
         jsonlines.open(paths["input_jsonl"]) as reader:

        for i, rec in enumerate(reader):
            if limit is not None and i >= limit:
                break
            if should_stop(counts, target_per_label):
                print("\nAll label quotas reached.", flush=True)
                break

            row_id = rec.get("row_id", i)
            img_id = C.safe_text(rec.get("Flickr30K_ID", ""))
            hypothesis = C.safe_text(rec.get("hypothesis", ""))
            gold = C.normalize_label(rec.get("gold", rec.get("annotator_label", "")))
            prediction = C.safe_text(rec.get("prediction", ""))
            final_label = C.normalize_label(rec.get("final_label", prediction))
            full_reason = C.safe_text(rec.get("reason", ""))
            atomic_facts = C.ensure_list_of_atoms(rec.get("atomic_facts", []))

            selected_candidate = C.safe_text(rec.get("selected_candidate", ""))
            selected_model = C.safe_text(rec.get("selected_model", ""))
            selected_method = C.safe_text(rec.get("selected_method", ""))
            selected_prompt = C.safe_text(rec.get("selected_prompt", ""))
            selected_candidate_label = C.normalize_label(
                rec.get("selected_candidate_label", ""))
            candidate_matches = C.bool_field(
                rec.get("candidate_matches_learned_label", None),
                default=selected_candidate_label == final_label)

            grounding_eligible = C.bool_field(
                rec.get("grounding_eligible", None),
                default=(candidate_matches
                         and final_label in C.GROUNDABLE_FINAL_LABELS))

            if not grounding_eligible:
                skipped += 1
                skip_reasons["not_grounding_eligible"] += 1
                write_skip(skipped_writer, rec, "not_grounding_eligible")
                continue

            if not candidate_matches:
                skipped += 1
                skip_reasons["candidate_not_matched"] += 1
                write_skip(skipped_writer, rec, "candidate_not_matched")
                continue

            if final_label not in C.GROUNDABLE_FINAL_LABELS:
                skipped += 1
                skip_reasons["final_label_not_entailment_or_contradiction"] += 1
                write_skip(skipped_writer, rec,
                           "final_label_not_entailment_or_contradiction")
                continue

            if counts[final_label] >= target_per_label:
                skipped += 1
                skip_reasons[f"quota_reached_{final_label}"] += 1
                write_skip(skipped_writer, rec, f"quota_reached_{final_label}")
                continue

            evidence_items = get_evidence_items_from_record(rec)
            if not evidence_items:
                skipped += 1
                skip_reasons["no_evidence_items"] += 1
                write_skip(skipped_writer, rec, "no_evidence_items")
                continue

            try:
                grounding_phrases = []
                for item_idx, item in enumerate(evidence_items, start=1):
                    atom = C.safe_text(item.get("atom", ""))
                    atom_label = C.normalize_label(item.get("atom_label", final_label))
                    vlm_reasoning = C.safe_text(item.get("vlm_reasoning", ""))

                    phrase_info = generate_grounding_phrase(
                        tokenizer, model, final_label, hypothesis, atom,
                        vlm_reasoning, full_reason, selected_model, selected_method)

                    grounding_phrases.append({
                        "phrase_index": item_idx,
                        "evidence_index": item.get("evidence_index", item_idx),
                        "final_prediction": final_label,
                        "final_label": final_label,
                        "prediction": prediction,
                        "score_prediction": C.safe_text(
                            rec.get("selected_candidate_score_prediction", "")),
                        "selected_candidate": selected_candidate,
                        "selected_model": selected_model,
                        "selected_method": selected_method,
                        "selected_prompt": selected_prompt,
                        "selected_candidate_label": selected_candidate_label,
                        "candidate_matches_learned_label": candidate_matches,
                        "atom": atom,
                        "atom_label": atom_label,
                        "vlm_reasoning": vlm_reasoning,
                        "evidence_source": C.safe_text(item.get("evidence_source", "")),
                        **phrase_info,
                    })

                grounding_phrases = C.deduplicate_phrase_items(grounding_phrases)

                num_groundable = sum(
                    1 for it in grounding_phrases
                    if it.get("can_ground", False)
                    and C.safe_text(it.get("phrase", "")).upper() != "NONE")

                if REQUIRE_GROUNDABLE_PHRASE and num_groundable <= 0:
                    skipped += 1
                    skip_reasons["no_groundable_phrase_after_generation"] += 1
                    write_skip(skipped_writer, rec,
                               "no_groundable_phrase_after_generation")
                    continue

                total_phrase_items += len(grounding_phrases)
                total_groundable += num_groundable

                out_writer.write({
                    "row_id": row_id,
                    "row_key_occurrence": rec.get("row_key_occurrence", None),
                    "Flickr30K_ID": img_id,
                    "annotator_label": rec.get("annotator_label", gold),
                    "gold": gold,
                    "hypothesis": hypothesis,
                    "atomic_facts": atomic_facts,
                    "final_label": final_label,
                    "prediction": prediction,
                    "selected_candidate": selected_candidate,
                    "selected_model": selected_model,
                    "selected_method": selected_method,
                    "selected_prompt": selected_prompt,
                    "selected_candidate_label": selected_candidate_label,
                    "candidate_matches_learned_label": candidate_matches,
                    "reason": full_reason,
                    "grounding_phrases": grounding_phrases,
                    "grounding_phrase": build_legacy_grounding_phrase(grounding_phrases),
                    "grounding_phrase_summary": grounding_phrase_summary(grounding_phrases),
                })
                out_writer._fp.flush()
                os.fsync(out_writer._fp.fileno())

                counts[final_label] += 1
                processed += 1

                if processed % PROGRESS_EVERY == 0:
                    print(f"  processed={processed} "
                          f"entailment={counts['entailment']} "
                          f"contradiction={counts['contradiction']} "
                          f"skipped={skipped}", flush=True)

            except Exception as e:
                failed += 1
                print(f"  ERROR at row {row_id} ({img_id}): {e}", flush=True)
                failed_writer.write({
                    "row_id": row_id,
                    "Flickr30K_ID": img_id,
                    "hypothesis": hypothesis,
                    "final_label": final_label,
                    "error": str(e),
                    "traceback": traceback.format_exc()[:4000],
                })
                failed_writer._fp.flush()
                continue

    print("")
    print(SEP)
    print("PHRASE EXTRACTION SUMMARY")
    print(SEP)
    print(f"Processed rows         : {processed}")
    print(f"  entailment           : {counts['entailment']}")
    print(f"  contradiction        : {counts['contradiction']}")
    print(f"Skipped rows           : {skipped}")
    print(f"Failed rows            : {failed}")
    print(f"Phrase items           : {total_phrase_items}")
    print(f"Groundable phrase items: {total_groundable}")
    print(DASH)
    for reason, count in skip_reasons.most_common():
        print(f"  {reason:<52} {count:>7}")
    print(f"\nWritten to {paths['output_jsonl']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--eval-name", default="AVE_learned_selection_evaluation_v3")
    ap.add_argument("--target-per-label", type=int, default=TARGET_PER_LABEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.split, args.eval_name, args.target_per_label, args.limit)


if __name__ == "__main__":
    main()