"""Baseline prediction: the full hypothesis judged directly."""

from src.prediction import prompts
from src.prediction.common import (
    normalize_label,
    normalize_predicted_label,
    normalize_text_field,
    extract_json_object,
    extract_label_from_free_text,
    clean_recovered_reason,
    recover_string_field,
    score_and_diagnose,
)

OUTPUT_NAME = "baseline"

FALLBACK_RULE = "if no JSON/no explicit label, use score_prediction"


# --- pipeline models: an unrecognised label becomes neutral ---

def parse_simple_plain(response):
    try:
        parsed = extract_json_object(response)
        label = normalize_label(parsed.get("label", "neutral"))
        reason = normalize_text_field(parsed.get("explanation", ""))
        reason = reason or "No explanation available."
        return label, reason, True, "", False, "json"
    except Exception as e:
        return "neutral", f"Parse error: {str(e)}", False, str(e), False, "json"


def parse_structured_plain(response):
    try:
        parsed = extract_json_object(response)
        label = normalize_label(parsed.get("label", "neutral"))
        evidence = normalize_text_field(parsed.get("visual_evidence", ""))
        reasoning = normalize_text_field(parsed.get("reasoning", ""))
        reason = f"{evidence} {reasoning}".strip() or "No reasoning available."
        return label, reason, True, "", False, "json"
    except Exception as e:
        return "neutral", f"Parse error: {str(e)}", False, str(e), False, "json"


# --- comparison models: JSON, then free text, then score_prediction ---

def parse_simple_recovery(response):
    try:
        parsed = extract_json_object(response)
        label = normalize_predicted_label(parsed.get("label", ""))
        reason = normalize_text_field(parsed.get("explanation", ""))
        reason = reason or clean_recovered_reason(
            response, ["explanation", "reasoning", "reason"]
        )

        if label is None:
            label = extract_label_from_free_text(response)
        if label is None:
            return None, reason, False, "No clear label found in JSON/free text", False, "score_fallback_needed"

        return label, reason, True, "", False, "json"

    except Exception as e:
        fallback_label = extract_label_from_free_text(response)
        reason = clean_recovered_reason(response, ["explanation", "reasoning", "reason"])
        if fallback_label is not None:
            return fallback_label, reason, True, "", True, "regex_recovery"
        return None, reason, False, str(e), True, "score_fallback_needed"


def parse_structured_recovery(response):
    try:
        parsed = extract_json_object(response)
        label = normalize_predicted_label(parsed.get("label", ""))

        evidence = normalize_text_field(parsed.get("visual_evidence", ""))
        reasoning = normalize_text_field(parsed.get("reasoning", ""))
        reason = f"{evidence} {reasoning}".strip()
        reason = reason or clean_recovered_reason(
            response, ["visual_evidence", "reasoning", "explanation", "reason"]
        )

        if label is None:
            label = extract_label_from_free_text(response)
        if label is None:
            return None, reason, False, "No clear label found in JSON/free text", False, "score_fallback_needed"

        return label, reason, True, "", False, "json"

    except Exception as e:
        fallback_label = extract_label_from_free_text(response)
        evidence = recover_string_field(response, ["visual_evidence"])
        reasoning = recover_string_field(response, ["reasoning", "explanation", "reason"])
        reason = f"{evidence} {reasoning}".strip()
        reason = reason or clean_recovered_reason(
            response, ["visual_evidence", "reasoning", "explanation", "reason"]
        )
        if fallback_label is not None:
            return fallback_label, reason, True, "", True, "regex_recovery"
        return None, reason, False, str(e), True, "score_fallback_needed"


VARIANTS = {
    "simple": {
        "prompt": prompts.direct_simple,
        "parse": {"plain": parse_simple_plain, "recovery": parse_simple_recovery},
        "strategy": "simple_direct",
    },
    "structured": {
        "prompt": prompts.direct_structured,
        "parse": {"plain": parse_structured_plain, "recovery": parse_structured_recovery},
        "strategy": "structured_direct",
    },
}


def prepare(record):
    return record


def score(adapter, image_ref, record, variant):
    spec = VARIANTS[variant]
    prompt_text = spec["prompt"](record["hypothesis"])

    raw = adapter.generate(image_ref, prompt_text)
    pred, reason, parse_ok, parse_error, fallback_used, parse_mode = \
        spec["parse"][adapter.parser](raw)

    scores, diagnostics = score_and_diagnose(adapter, image_ref, prompt_text)
    score_prediction = max(scores, key=scores.get)

    # Parser-only fallback. Prompt and likelihood scoring are unchanged.
    if pred is None:
        pred = score_prediction
        parse_ok = False
        fallback_used = True
        parse_mode = "score_fallback"

    result = {
        "strategy": spec["strategy"],
        "scores": scores,
        "prediction": pred,
        "score_prediction": score_prediction,
        "reason": reason,
        "parse_ok": parse_ok,
        "parse_error": parse_error,
        "fallback_used": fallback_used,
        "parse_mode": parse_mode,
        "raw_response": raw,
        "parser": adapter.parser,
        "hf_id": adapter.hf_id,
        "vlm_key": adapter.vlm_key,
    }
    result.update(diagnostics)
    return result


def build_record(record, result):
    inner = {
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
        "parse_ok": result["parse_ok"],
        "parse_error": result["parse_error"],
    }

    details = {"strategy": result["strategy"]}

    if result.get("parser") == "recovery":
        inner["fallback_used"] = result["fallback_used"]
        inner["parse_mode"] = result["parse_mode"]
        details["model_id"] = result["hf_id"]
        details["backend"] = result["vlm_key"]
        details["fallback_rule"] = FALLBACK_RULE
        details["raw_output"] = result["raw_response"]

    inner["strategy_details"] = details

    return {
        "Flickr30K_ID": record["img_id"],
        "annotator_label": record["gold"],
        "hypothesis": record["hypothesis"],
        "full_hypothesis_results": inner,
    }