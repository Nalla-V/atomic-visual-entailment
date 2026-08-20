"""Independent atomic prediction: each atom judged separately against the image."""

from src.prediction import prompts
from src.prediction.common import (
    normalize_label,
    normalize_text_field,
    extract_json_object,
    score_and_diagnose,
)

OUTPUT_NAME = "atomic"


def parse_simple(response):
    try:
        parsed = extract_json_object(response)
        label = normalize_label(parsed.get("label", "neutral"))
        reason = normalize_text_field(parsed.get("explanation", ""))
        reason = reason or "No explanation available."
        return label, reason, True, ""
    except Exception as e:
        return "neutral", f"Parse error: {str(e)}", False, str(e)


def parse_structured(response):
    try:
        parsed = extract_json_object(response)
        label = normalize_label(parsed.get("label", "neutral"))
        evidence = normalize_text_field(parsed.get("visual_evidence", ""))
        reasoning = normalize_text_field(parsed.get("reasoning", ""))
        reason = f"{evidence} {reasoning}".strip()
        reason = reason or "No reasoning available."
        return label, reason, True, ""
    except Exception as e:
        return "neutral", f"Parse error: {str(e)}", False, str(e)


VARIANTS = {
    "simple": {
        "prompt": prompts.direct_simple,
        "parse": parse_simple,
        "strategy": "simple_direct",
    },
    "structured": {
        "prompt": prompts.direct_structured,
        "parse": parse_structured,
        "strategy": "structured_direct",
    },
}


def prepare(record):
    """Decomposition falls back to the hypothesis as a single atom, so an empty
    list means a malformed row. Recover the same way."""
    if not record["atoms"]:
        record["atoms"] = [record["hypothesis"]]
        record["atoms_fallback"] = True
    return record


def score(adapter, image_ref, record, variant):
    """One result per atom, so this returns a list."""
    spec = VARIANTS[variant]
    results = []

    for atom_text in record["atoms"]:
        prompt_text = spec["prompt"](atom_text)

        raw = adapter.generate(image_ref, prompt_text)
        pred, reason, parse_ok, parse_error = spec["parse"](raw)

        scores, diagnostics = score_and_diagnose(adapter, image_ref, prompt_text)
        score_prediction = max(scores, key=scores.get)

        result = {
            "strategy": spec["strategy"],
            "scores": scores,
            "prediction": pred,
            "score_prediction": score_prediction,
            "reason": reason,
            "parse_ok": parse_ok,
            "parse_error": parse_error,
            "raw_response": raw,
        }
        result.update(diagnostics)
        results.append((atom_text, result))

    return {"per_atom": results,
            "parse_ok": all(r["parse_ok"] for _, r in results),
            "parse_error": "; ".join(r["parse_error"] for _, r in results if r["parse_error"]),
            "raw_response": "\n---\n".join(r["raw_response"] for _, r in results)}


def build_record(record, result):
    atom_entries = []

    for atom_text, r in result.get("per_atom", []):
        atom_entries.append({
            "atom_text": atom_text,
            "initial_results": {
                "scores": r["scores"],
                "initial_prediction": r["prediction"],
                "initial_score_prediction": r["score_prediction"],
                "confidence_score": r["confidence_score"],
                "margin": r["margin"],
                "entropy": r["entropy"],
                "normalized_entropy": r["normalized_entropy"],
                "top_label": r["top_label"],
                "second_label": r["second_label"],
                "initial_reason": r["reason"],
                "parse_ok": r["parse_ok"],
                "parse_error": r["parse_error"],
                "strategy_details": {
                    "strategy": r["strategy"],
                },
            },
        })

    return {
        "Flickr30K_ID": record["img_id"],
        "annotator_label": record["gold"],
        "hypothesis": record["hypothesis"],
        "atomic_facts": atom_entries,
    }