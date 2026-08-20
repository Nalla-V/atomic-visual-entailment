"""Self-decomposition: the VLM decomposes the hypothesis and labels each atom
in one pass. The reported prediction is the rule-based aggregation over those
atom labels, not the model's own final label."""

from src.prediction import prompts
from src.prediction.common import (
    normalize_label,
    normalize_text_field,
    extract_json_object,
    score_and_diagnose,
)

OUTPUT_NAME = "self_decompose"


def aggregate_from_atom_labels(atom_labels):
    """contradiction wins, then neutral, else entailment."""
    atom_labels = [normalize_label(x) for x in atom_labels]
    if not atom_labels:
        return "neutral"
    if "contradiction" in atom_labels:
        return "contradiction"
    if "neutral" in atom_labels:
        return "neutral"
    return "entailment"


def parse(response):
    try:
        parsed = extract_json_object(response)
        model_label = normalize_label(parsed.get("label", "neutral"))

        bridge_reasoning = normalize_text_field(
            parsed.get("bridge_reasoning", parsed.get("explanation", ""))
        )

        atoms = parsed.get("decomposed_atoms", [])
        if not isinstance(atoms, list):
            atoms = []

        cleaned_atoms = []
        for item in atoms:
            if not isinstance(item, dict):
                continue

            atom_text = normalize_text_field(
                item.get("atom", item.get("claim", item.get("fact", "")))
            )
            atom_label = normalize_label(
                item.get("label", item.get("status", "neutral"))
            )
            atom_reason = normalize_text_field(
                item.get("reason",
                         item.get("evidence", item.get("visible_evidence", "")))
            )

            if atom_text:
                cleaned_atoms.append({
                    "atom": atom_text,
                    "label": atom_label,
                    "reason": atom_reason,
                })

        rule_label = aggregate_from_atom_labels(
            [atom["label"] for atom in cleaned_atoms]
        )

        if not bridge_reasoning:
            bridge_reasoning = "No bridge reasoning available."

        return model_label, rule_label, bridge_reasoning, cleaned_atoms, True, ""

    except Exception as e:
        return "neutral", "neutral", f"Parse error: {str(e)}", [], False, str(e)


VARIANTS = {
    "simple": {"prompt": prompts.selfdecompose_simple},
    "structured": {"prompt": prompts.selfdecompose_structured},
}


def prepare(record):
    return record


def score(adapter, image_ref, record, variant):
    spec = VARIANTS[variant]
    prompt_text = spec["prompt"](record["hypothesis"])

    raw = adapter.generate(image_ref, prompt_text)
    (model_prediction, rule_prediction, reason,
     decomposed_atoms, parse_ok, parse_error) = parse(raw)

    scores, diagnostics = score_and_diagnose(adapter, image_ref, prompt_text)
    score_prediction = max(scores, key=scores.get)

    result = {
        "strategy": f"{record['vlm']}_self_decompose_{variant}",
        "prompt_style": variant,
        "prediction": rule_prediction,
        "model_prediction": model_prediction,
        "score_prediction": score_prediction,
        "aggregation_mismatch": model_prediction != rule_prediction,
        "scores": scores,
        "reason": reason,
        "decomposed_atoms": decomposed_atoms,
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
        "self_decompose_results": {
            "scores": result["scores"],
            "prediction": result["prediction"],
            "model_prediction": result["model_prediction"],
            "score_prediction": result["score_prediction"],
            "aggregation_mismatch": result["aggregation_mismatch"],
            "confidence_score": result["confidence_score"],
            "margin": result["margin"],
            "entropy": result["entropy"],
            "normalized_entropy": result["normalized_entropy"],
            "top_label": result["top_label"],
            "second_label": result["second_label"],
            "reason": result["reason"],
            "decomposed_atoms": result["decomposed_atoms"],
            "parse_ok": result["parse_ok"],
            "parse_error": result["parse_error"],
            "strategy_details": {
                "strategy": result["strategy"],
                "prompt_style": result["prompt_style"],
                "input_to_model": "image + hypothesis",
                "self_decomposition": True,
                "external_atoms_used": False,
            },
        },
    }