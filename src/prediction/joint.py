"""Joint atomic prediction: all atoms judged together in one pass."""

from src.prediction import prompts
from src.prediction.common import (
    normalize_label,
    normalize_text_field,
    extract_json_object,
    score_and_diagnose,
)

OUTPUT_NAME = "atomic_joint"


def parse_simple(response):
    try:
        parsed = extract_json_object(response)
        label = normalize_label(parsed.get("label", "neutral"))
        reason = normalize_text_field(parsed.get("explanation", ""))
        reason = reason or "No explanation available."
        return label, reason, [], True, ""
    except Exception as e:
        return "neutral", f"Parse error: {str(e)}", [], False, str(e)


def parse_structured(response):
    try:
        parsed = extract_json_object(response)
        label = normalize_label(parsed.get("label", "neutral"))
        bridge_reasoning = normalize_text_field(parsed.get("bridge_reasoning", ""))

        atom_observations = parsed.get("atom_observations", [])
        if not isinstance(atom_observations, list):
            atom_observations = []

        cleaned = []
        for obs in atom_observations:
            if not isinstance(obs, dict):
                continue
            cleaned.append({
                "atom": normalize_text_field(obs.get("atom", "")),
                "label": normalize_label(obs.get("label", obs.get("status", "neutral"))),
                "reason": normalize_text_field(obs.get("reason", "")),
            })

        reason = bridge_reasoning or "No bridge reasoning available."
        return label, reason, cleaned, True, ""
    except Exception as e:
        return "neutral", f"Parse error: {str(e)}", [], False, str(e)


VARIANTS = {
    "simple": {
        "prompt": prompts.joint_simple,
        "parse": parse_simple,
        "strategy": "atomic_joint_atoms_only_simple_direct",
    },
    "structured": {
        "prompt": prompts.joint_structured,
        "parse": parse_structured,
        "strategy": "atomic_joint_atoms_only_structured_direct",
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
    spec = VARIANTS[variant]
    prompt_text = spec["prompt"](record["atoms"])

    raw = adapter.generate(image_ref, prompt_text)
    pred, reason, observations, parse_ok, parse_error = spec["parse"](raw)

    scores, diagnostics = score_and_diagnose(adapter, image_ref, prompt_text)
    score_prediction = max(scores, key=scores.get)

    result = {
        "strategy": spec["strategy"],
        "scores": scores,
        "prediction": pred,
        "score_prediction": score_prediction,
        "reason": reason,
        "atom_observations": observations,
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "raw_response": raw,
    }
    result.update(diagnostics)
    return result


def build_record(record, result):
    return {
        "Flickr30K_ID": record["img_id"],
        "annotator_label": record["gold"],
        "hypothesis": record["hypothesis"],
        "atomic_facts": record["atoms"],
        "joint_atom_results": {
            "scores": result["scores"],
            "prediction": result["prediction"],
            "score_prediction": result["score_prediction"],
            "confidence_score": result["confidence_score"],
            "margin": result["margin"],
            "entropy": result["entropy"],
            "normalized_entropy": result["normalized_entropy"],
            "top_label": result["top_label"],
            "second_label": result["second_label"],
            "reason": result["reason"],
            "atom_observations": result["atom_observations"],
            "parse_ok": result["parse_ok"],
            "parse_error": result["parse_error"],
            "strategy_details": {
                "strategy": result["strategy"],
                "input_to_model": "image + atomic_facts_only",
                "hypothesis_passed_to_model": False,
            },
        },
    }