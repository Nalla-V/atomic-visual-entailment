"""Shared helpers for the prediction stage: text cleaning, label handling,
JSON extraction, and confidence diagnostics."""

import json
import math
import os
import re

from src import config

def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(v).strip() for v in x if str(v).strip())
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()


def normalize_label(label_text):
    text = safe_text(label_text).lower().strip()
    for label in config.FINAL_LABELS:
        if label in text:
            return label
    return ""


def normalize_text_field(x):
    if isinstance(x, list):
        return " ".join(str(v).strip() for v in x if str(v).strip())
    return str(x).strip()


def ensure_list_of_atoms(atoms):
    if not isinstance(atoms, list):
        return []
    out = []
    for atom in atoms:
        if isinstance(atom, str):
            text = atom.strip()
        elif isinstance(atom, dict):
            text = safe_text(atom.get("atom_text", atom.get("text", atom.get("atom", ""))))
        else:
            text = ""
        if text:
            out.append(text)
    return out


def format_atoms(atoms):
    return "\n".join(f"{i + 1}. {atom}" for i, atom in enumerate(atoms))


def extract_json_object(text):
    text = safe_text(text)
    text = text.replace("```json", "```").replace("```JSON", "```")
    text = re.sub(r"```(.*?)```", r"\1", text, flags=re.DOTALL).strip()

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found")

    raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        repaired = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(repaired)


def normalize_predicted_label(label_text):
    """Returns None when no clear label is found, so the caller can fall back
    to score_prediction rather than silently writing neutral."""
    text = safe_text(label_text).lower().strip()
    if not text:
        return None
    if text in config.FINAL_LABELS:
        return text
    if "contradiction" in text or "contradicts" in text or "contradictory" in text:
        return "contradiction"
    if "entailment" in text or "entailed" in text or "entails" in text:
        return "entailment"
    if "neutral" in text:
        return "neutral"
    if "clear incompatible" in text or "clearly conflicts" in text or "conflict" in text:
        return "contradiction"
    if "clearly supports" in text or "supported by the image" in text:
        return "entailment"
    if "insufficient" in text or "not enough evidence" in text or "uncertain" in text:
        return "neutral"
    return None


def strip_prompt_echo(text):
    text = safe_text(text)
    if not text:
        return ""

    text = text.replace("```json", "```").replace("```JSON", "```")
    text = re.sub(r"```(.*?)```", r"\1", text, flags=re.DOTALL).strip()

    markers = [
        "Return ONLY JSON in this format:",
        "Return ONLY valid JSON in this exact format:",
        "Choose exactly one label from:",
    ]
    for marker in markers:
        if marker in text:
            text = text.split(marker)[-1].strip()

    text = re.sub(r'["\']?label["\']?\s*[:=]\s*["\']?', "label: ", text, flags=re.IGNORECASE)
    text = re.sub(r'["\']?explanation["\']?\s*[:=]\s*["\']?', "explanation: ", text, flags=re.IGNORECASE)
    text = re.sub(r'["\']?visual_evidence["\']?\s*[:=]\s*["\']?', "visual evidence: ", text, flags=re.IGNORECASE)
    text = re.sub(r'["\']?reasoning["\']?\s*[:=]\s*["\']?', "reasoning: ", text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()


def extract_label_from_free_text(response):
    text = strip_prompt_echo(response)
    low = text.lower()

    m = re.search(r"\blabel\s*[:=-]\s*(entailment|neutral|contradiction)\b", low)
    if m:
        return m.group(1)

    m = re.search(
        r"\b(?:final\s+label|final\s+answer|answer|prediction)\s*(?:is|:|-)?\s*"
        r"(entailment|neutral|contradiction)\b",
        low,
    )
    if m:
        return m.group(1)

    found = [label for label in config.FINAL_LABELS if re.search(rf"\b{label}\b", low)]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        positions = {label: low.find(label) for label in found if low.find(label) >= 0}
        return min(positions, key=positions.get)

    return normalize_predicted_label(low)


def clean_reason_from_response(response):
    text = strip_prompt_echo(response)
    text = re.sub(
        r"^\s*label\s*[:=-]\s*(entailment|neutral|contradiction)\s*[,.;-]*\s*",
        "", text, flags=re.IGNORECASE,
    )
    for k in ["explanation:", "reasoning:", "visual evidence:"]:
        idx = text.lower().find(k)
        if idx >= 0:
            text = text[idx + len(k):].strip()
    return re.sub(r"\s+", " ", text).strip() or "No explanation available."


def recover_string_field(text, field_names):
    raw = strip_prompt_echo(text)
    for field in field_names:
        m = re.search(
            rf'"{field}"\s*:\s*"(?P<val>[^"\\]*(?:\\.[^"\\]*)*)"',
            raw, flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            return normalize_text_field(m.group("val"))

        m = re.search(rf'"{field}"\s*:\s*"(?P<val>.*)$', raw,
                      flags=re.IGNORECASE | re.DOTALL)
        if m:
            val = m.group("val")
            val = re.split(
                r'"\s*,\s*"(?:label|explanation|visual_evidence|reasoning|reason)"\s*:',
                val, maxsplit=1, flags=re.IGNORECASE,
            )[0]
            val = re.sub(r'[}\]]+\s*$', '', val).strip().strip('"')
            return normalize_text_field(val)
    return ""


def clean_recovered_reason(text, fields):
    reason = recover_string_field(text, fields)
    if reason:
        return reason
    reason = clean_reason_from_response(text)
    reason = re.sub(r'[}\]]+\s*$', '', reason).strip().strip('"')
    return reason or "No explanation available."


def softmax_dict(raw_scores):
    max_score = max(raw_scores.values())
    exp_scores = {k: math.exp(v - max_score) for k, v in raw_scores.items()}
    z = sum(exp_scores.values())
    return {k: round(exp_scores[k] / z, 6) for k in config.FINAL_LABELS}


def confidence_diagnostics(scores):
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_prob = ordered[0]
    second_label, second_prob = ordered[1]

    entropy = -sum(p * math.log(p + 1e-12) for p in scores.values())
    normalized_entropy = entropy / math.log(len(config.FINAL_LABELS))

    return {
        "confidence_score": round(top_prob, 6),
        "margin": round(top_prob - second_prob, 6),
        "entropy": round(entropy, 6),
        "normalized_entropy": round(normalized_entropy, 6),
        "top_label": top_label,
        "second_label": second_label,
    }


def score_and_diagnose(adapter, image_ref, prompt_text):
    raw = adapter.score_labels(image_ref, prompt_text)
    scores = softmax_dict(raw)
    return scores, confidence_diagnostics(scores)


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)