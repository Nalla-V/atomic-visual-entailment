"""Phrase cleaning and validation for grounding phrase extraction."""

import json
import re

from src import config

LABELS = config.FINAL_LABELS
GROUNDABLE_FINAL_LABELS = ["entailment", "contradiction"]

# Words that signal reasoning rather than something visible.
BAD_WORDS = [
    "because", "therefore", "entails", "supports", "indicates", "suggests",
    "likely", "probably", "hypothesis", "prediction", "reason",
    "cannot determine", "not enough", "insufficient", "unclear",
]

ABSENCE_PATTERNS = [
    "no ", "not visible", "not shown", "cannot see", "can't see", "absent",
    "missing", "there is no", "there are no", "does not show", "doesn't show",
    "not present",
]

# Clauses after these markers describe what is absent, not what is visible.
NEGATION_SPLITS = [
    r",\s*not\b", r"\bnot\b", r"\binstead of\b", r"\brather than\b",
    r"\bwithout\b", r"\bbut no\b", r"\bbut not\b",
]


def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(v).strip() for v in x if str(v).strip())
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()


def normalize_label(x):
    text = safe_text(x).lower().strip()
    if text in LABELS:
        return text
    if "entailment" in text or "entailed" in text or "support" in text:
        return "entailment"
    if "contradiction" in text or "contradict" in text or "conflict" in text:
        return "contradiction"
    if "neutral" in text or "uncertain" in text or "insufficient" in text:
        return "neutral"
    for lab in LABELS:
        if lab in text:
            return lab
    return "neutral"


def ensure_list_of_atoms(atoms):
    if not isinstance(atoms, list):
        return []
    out = []
    for atom in atoms:
        if isinstance(atom, str):
            text = safe_text(atom)
        elif isinstance(atom, dict):
            text = safe_text(atom.get("atom", atom.get("atom_text",
                             atom.get("claim", atom.get("fact", "")))))
        else:
            text = ""
        if text:
            out.append(text)
    return out


def clean_model_output(text):
    text = safe_text(text)
    text = text.replace("```json", "```").replace("```JSON", "```")
    text = re.sub(r"```(.*?)```", r"\1", text, flags=re.DOTALL)
    return text.strip()


def extract_json_object(text):
    text = clean_model_output(text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    raw = m.group(0)
    try:
        parsed = json.loads(raw)
    except Exception:
        try:
            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))
        except Exception:
            return None
    return parsed if isinstance(parsed, dict) else None


def bool_field(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = safe_text(value).lower().strip()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return default


def phrase_is_bad(phrase):
    phrase = safe_text(phrase).strip()
    if not phrase:
        return True
    if phrase.upper() == "NONE":
        return False

    p = phrase.lower()
    if len(p.split()) > 8:
        return True
    if any(w in p for w in BAD_WORDS):
        return True
    # Grounding DINO does not localise negated phrases well.
    if re.search(r"\b(no|not|without|absent|missing)\b", p):
        return True
    return False


def clean_phrase(text):
    if text is None:
        return ""

    text = clean_model_output(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = lines[0] if lines else ""

    for prefix in ["grounding phrase:", "phrase:", "answer:", "output:"]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()

    text = text.replace('"', "").replace("'", "")
    text = text.rstrip(".").rstrip(",")
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return "NONE" if text.upper() == "NONE" else text


def clean_match_phrase(text):
    text = clean_phrase(text)
    if not text or text.upper() == "NONE":
        return "NONE"

    words = text.split()
    # match_phrase stays short because it is used for entity matching.
    if len(words) > 4:
        text = " ".join(words[:4])

    return "NONE" if phrase_is_bad(text) else text


def fallback_match_phrase(phrase):
    phrase = clean_phrase(phrase)
    if not phrase or phrase.upper() == "NONE":
        return "NONE"
    words = phrase.split()
    if not words:
        return "NONE"
    # First one to three words of the grounding phrase. The model-provided
    # match_phrase is preferred whenever available.
    return " ".join(words[:min(3, len(words))])


def is_absence_only(reason):
    text = safe_text(reason).lower()
    return any(p in text for p in ABSENCE_PATTERNS)


def positive_visual_clause(text):
    """Keep the visible positive alternative and drop the negated clause.

    "The toddler is sitting at a desk indoors, not sleeping outside."
    becomes "The toddler is sitting at a desk indoors".
    """
    text = safe_text(text)
    lowered = text.lower()

    cut_positions = []
    for pat in NEGATION_SPLITS:
        m = re.search(pat, lowered)
        if m:
            cut_positions.append(m.start())

    if cut_positions:
        text = text[:min(cut_positions)].strip(" ,.;:")

    return text.strip()


def fallback_phrase(atom, reason, final_label):
    """Build a phrase from the reasoning text when the model gives nothing usable."""
    final_label = normalize_label(final_label)
    reason = positive_visual_clause(reason)

    if final_label == "contradiction" and is_absence_only(reason):
        # Only absence is described, so there is no visible object to localise.
        return "NONE"

    text = (reason if reason else atom).lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(
        r"\b(the|a|an|this|that|there|image|shows|showing|visible|clearly|likely"
        r"|with|which|supports|statement|claim|hypothesis|is|are|was|were|be"
        r"|being|to|of|in|on|at|and|or)\b", " ", text)
    text = re.sub(r"\b(no|not|without|absent|missing|cannot|can't|doesn|doesnt"
                  r"|don't|dont)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    phrase = " ".join(words[:6]) if words else "NONE"

    return "NONE" if phrase_is_bad(phrase) else phrase


def deduplicate_phrase_items(items):
    seen, out = set(), []
    for item in items:
        key = (safe_text(item.get("phrase", "")).lower().strip(),
               safe_text(item.get("atom", "")).lower().strip(),
               safe_text(item.get("evidence_source", "")).lower().strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out