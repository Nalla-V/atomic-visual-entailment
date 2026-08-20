"""Localise grounding phrases with Grounding DINO.

    python -m src.grounding.detect --split dev

Reads the phrase file, queries Grounding DINO for each groundable phrase,
suppresses near-duplicate boxes, keeps the top boxes, and writes a detection
record per row.
"""

import argparse
import json
import os
import re
import sys
import traceback

import jsonlines
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import config

MODEL_ID = "IDEA-Research/grounding-dino-base"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

MAX_BOXES_TO_DRAW = 3
MAX_PHRASES_PER_ROW = 5

REPORT_TARGET_WIDTH = 900
REPORT_BOX_WIDTH = 6
REPORT_BADGE_SIZE = 36
REPORT_BORDER_WIDTH = 2

DUPLICATE_COORD_TOLERANCE = 3.0
DUPLICATE_IOU_THRESHOLD = 0.92
DUPLICATE_CONTAINMENT_THRESHOLD = 0.95
DUPLICATE_SIZE_RATIO_THRESHOLD = 0.80

PROGRESS_EVERY = 10

FINAL_LABELS = config.FINAL_LABELS
GROUNDABLE_FINAL_LABELS = ["entailment", "contradiction"]

SEP = "=" * 120


# ============================================================
# 1. PATHS
# ============================================================

def build_paths(split, eval_name):
    if split not in {"dev", "test"}:
        raise ValueError("split must be either 'dev' or 'test'")

    dataset_dir = os.path.join(config.OUTPUT_DIR, f"{split}_dataset")
    eval_dir = os.path.join(dataset_dir, eval_name)
    grounding_dir = os.path.join(eval_dir, "grounding_dino_v1")

    image_output_dir = os.path.join(
        grounding_dir, f"annotated_images_ave_ls_final_{split}")
    report_image_output_dir = os.path.join(
        grounding_dir, f"annotated_images_report_ave_ls_final_{split}")

    for d in [grounding_dir, image_output_dir, report_image_output_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "split": split,
        "grounding_dir": grounding_dir,
        "input_jsonl": os.path.join(
            grounding_dir, f"grounding_phrase_ave_ls_final_{split}.jsonl"),
        "output_jsonl": os.path.join(
            grounding_dir, f"grounding_dino_boxes_ave_ls_final_{split}.jsonl"),
        "failed_jsonl": os.path.join(
            grounding_dir, f"grounding_dino_ave_ls_final_{split}_failed_rows.jsonl"),
        "image_output_dir": image_output_dir,
        "report_image_output_dir": report_image_output_dir,
    }


# ============================================================
# 2. TEXT HELPERS
# ============================================================

def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return " ".join(str(v).strip() for v in x if str(v).strip())
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()


def clean_spacing(text):
    """Cleanup for one-line text. Collapses runs of spaces but not newlines."""
    text = safe_text(text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])([A-Za-z])", r"\1 \2", text)
    return text


def clean_multiline_text(text):
    """Cleanup for panel text, preserving line breaks between list items."""
    lines = [clean_spacing(line) for line in safe_text(text).split("\n")]
    return "\n".join(line for line in lines if line)


def shorten_text(text, max_chars=420):
    text = clean_spacing(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + "..."


def shorten_multiline_text(text, max_chars=420):
    text = clean_multiline_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3].rstrip() + "..."


def normalize_label(x):
    text = safe_text(x).lower().strip()
    if text in FINAL_LABELS:
        return text
    for label in FINAL_LABELS:
        if label in text:
            return label
    return text


def load_image(flickr_id):
    image_path = os.path.join(config.IMAGE_DIR, f"{flickr_id}.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB"), image_path


def make_text_query(phrase):
    phrase = phrase.strip()
    if not phrase.endswith("."):
        phrase += "."
    return phrase


def get_color(label):
    label = normalize_label(label)
    if label == "entailment":
        return (0, 150, 60)
    if label == "contradiction":
        return (220, 30, 30)
    return (210, 160, 0)


def safe_filename(text):
    text = safe_text(text)
    keep = [ch if (ch.isalnum() or ch in ["_", "-", "."]) else "_" for ch in text]
    return "".join(keep)[:180]


def phrase_key(phrase):
    phrase = safe_text(phrase).lower().strip()
    return re.sub(r"\s+", " ", phrase)


# ============================================================
# 3. ATOMIC FACT HELPERS
# ============================================================

def ensure_atomic_facts(atomic_facts):
    if not isinstance(atomic_facts, list):
        return []

    facts = []
    for item in atomic_facts:
        if isinstance(item, str):
            text = clean_spacing(item)
        elif isinstance(item, dict):
            text = clean_spacing(item.get("atom_text",
                                 item.get("atom", item.get("text", ""))))
        else:
            text = ""
        if text:
            facts.append(text)
    return facts


def normalize_atom_key(text):
    text = clean_spacing(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_atomic_fact_mapping(atomic_facts, phrase_items):
    """Number the atoms A1, A2, ... so the panel can cross-reference them."""
    display_atoms, seen = [], set()

    for atom in atomic_facts:
        key = normalize_atom_key(atom)
        if key and key not in seen:
            seen.add(key)
            display_atoms.append(clean_spacing(atom))

    for item in phrase_items:
        atom = clean_spacing(item.get("atom", ""))
        key = normalize_atom_key(atom)
        if key and key not in seen:
            seen.add(key)
            display_atoms.append(atom)

    atom_to_ref = {normalize_atom_key(atom): f"A{idx}"
                   for idx, atom in enumerate(display_atoms, start=1)}

    return display_atoms, atom_to_ref


def get_atom_ref(item, atom_to_ref):
    return atom_to_ref.get(normalize_atom_key(safe_text(item.get("atom", ""))), "")


def build_atomic_facts_text(display_atoms):
    if not display_atoms:
        return "No atomic facts available."
    return "\n".join(f"A{idx}. {clean_spacing(atom)}"
                     for idx, atom in enumerate(display_atoms, start=1))


# ============================================================
# 4. PHRASE ITEMS
# ============================================================

def normalize_phrase_item(item):
    phrase = clean_spacing(item.get("phrase", ""))
    can_ground = bool(item.get("can_ground", True))
    if phrase.upper() == "NONE":
        can_ground = False

    return {
        "phrase_index": item.get("phrase_index", None),
        "evidence_index": item.get("evidence_index", None),
        "phrase": phrase,
        "match_phrase": clean_spacing(item.get("match_phrase", "")),
        "can_ground": can_ground,
        "atom": clean_spacing(item.get("atom", item.get("selected_atom", ""))),
        "atom_label": normalize_label(
            item.get("atom_label", item.get("selected_atom_label", ""))),
        "vlm_reasoning": clean_spacing(
            item.get("vlm_reasoning", item.get("selected_atom_reason", ""))),
        "evidence_source": safe_text(item.get("evidence_source", "")),
        "final_prediction": normalize_label(
            item.get("final_prediction", item.get("final_label", ""))),
        "final_label": normalize_label(
            item.get("final_label", item.get("final_prediction", ""))),
        "prediction": normalize_label(
            item.get("prediction", item.get("final_prediction", ""))),
        "score_prediction": normalize_label(item.get("score_prediction", "")),
        "selected_candidate": safe_text(item.get("selected_candidate", "")),
        "selected_model": safe_text(item.get("selected_model", "")),
        "selected_method": safe_text(item.get("selected_method", "")),
        "selected_prompt": safe_text(item.get("selected_prompt", "")),
        "selected_candidate_label": normalize_label(
            item.get("selected_candidate_label", "")),
        "candidate_matches_learned_label": bool(
            item.get("candidate_matches_learned_label", True)),
        "phrase_source": safe_text(item.get("phrase_source", "")),
        "phrase_model": safe_text(item.get("phrase_model", "")),
        "raw_model_output": safe_text(item.get("raw_model_output", "")),
    }


def get_grounding_phrase_items(rec):
    items = []
    multi = rec.get("grounding_phrases", [])
    if isinstance(multi, list) and multi:
        for item in multi:
            if isinstance(item, dict):
                items.append(normalize_phrase_item(item))
    else:
        single = rec.get("grounding_phrase", {})
        if isinstance(single, dict):
            items.append(normalize_phrase_item(single))

    filtered, seen_phrases = [], set()
    for item in items:
        phrase = safe_text(item.get("phrase", ""))
        if not item.get("can_ground", False):
            continue
        if not phrase or phrase.upper() == "NONE":
            continue
        key = phrase_key(phrase)
        if key in seen_phrases:
            continue
        seen_phrases.add(key)
        filtered.append(item)

    return filtered[:MAX_PHRASES_PER_ROW]


# ============================================================
# 5. BOX FILTERING
# ============================================================

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_intersection(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_iou(box_a, box_b):
    inter = box_intersection(box_a, box_b)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def box_containment_overlap(box_a, box_b):
    inter = box_intersection(box_a, box_b)
    smaller = min(box_area(box_a), box_area(box_b))
    return inter / smaller if smaller > 0 else 0.0


def box_size_ratio(box_a, box_b):
    area_a, area_b = box_area(box_a), box_area(box_b)
    larger, smaller = max(area_a, area_b), min(area_a, area_b)
    return smaller / larger if larger > 0 else 0.0


def boxes_are_near_identical(box_a, box_b, tolerance=DUPLICATE_COORD_TOLERANCE):
    """Grounding DINO may return the same box for several phrases."""
    if len(box_a) != 4 or len(box_b) != 4:
        return False
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(box_a, box_b))


def should_suppress_duplicate_box(candidate, chosen):
    """Suppress near-duplicates only. Deliberately conservative, so that
    different objects that merely overlap are kept."""
    candidate_box, chosen_box = candidate["box"], chosen["box"]

    if boxes_are_near_identical(candidate_box, chosen_box):
        return True
    if box_iou(candidate_box, chosen_box) >= DUPLICATE_IOU_THRESHOLD:
        return True
    if (box_containment_overlap(candidate_box, chosen_box) >= DUPLICATE_CONTAINMENT_THRESHOLD
            and box_size_ratio(candidate_box, chosen_box) >= DUPLICATE_SIZE_RATIO_THRESHOLD):
        return True
    return False


def select_unique_detections(detections):
    """Remove near-duplicate boxes while preserving score order, so repeated
    boxes do not occupy multiple top-K slots."""
    if not detections:
        return []

    unique = []
    for det in detections:
        if not any(should_suppress_duplicate_box(det, chosen) for chosen in unique):
            unique.append(det)
    return unique


# ============================================================
# 6. FONTS AND TEXT DRAWING
# ============================================================

def load_font(size, bold=False):
    try:
        path = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
                else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    except Exception:
        pass
    return ImageFont.load_default()


FONT_TITLE = load_font(60, bold=True)
FONT_HEADER = load_font(18, bold=True)
FONT_BODY = load_font(16, bold=False)
FONT_BOX = load_font(16, bold=True)


def text_width(draw, text, font):
    try:
        return int(draw.textlength(text, font=font))
    except Exception:
        return draw.textbbox((0, 0), text, font=font)[2]


def wrap_one_line(draw, text, font, max_width):
    text = safe_text(text)
    if not text:
        return [""]

    lines, current = [], ""
    for word in text.split():
        candidate = word if not current else current + " " + word
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    return lines if lines else [""]


def draw_wrapped_text(draw, xy, text, font, fill, max_width,
                      line_gap=6, paragraph_gap=8):
    x, y = xy
    text = safe_text(text)
    paragraphs = text.split("\n") if text else [""]

    for p_idx, paragraph in enumerate(paragraphs):
        for line in wrap_one_line(draw, paragraph, font, max_width):
            draw.text((x, y), line, font=font, fill=fill)
            try:
                bbox = draw.textbbox((x, y), line, font=font)
                line_h = bbox[3] - bbox[1]
            except Exception:
                line_h = 16
            y += line_h + line_gap
        if p_idx < len(paragraphs) - 1:
            y += paragraph_gap

    return y


def draw_centered_wrapped_text(draw, x_left, x_right, y, text, font, fill, line_gap=4):
    max_width = x_right - x_left
    for line in wrap_one_line(draw, text, font, max_width):
        w = text_width(draw, line, font)
        x = x_left + max(0, (max_width - w) / 2)
        draw.text((x, y), line, font=font, fill=fill)
        try:
            bbox = draw.textbbox((x, y), line, font=font)
            line_h = bbox[3] - bbox[1]
        except Exception:
            line_h = 18
        y += line_h + line_gap
    return y


def draw_section(draw, x, y, title, body, max_width, title_color):
    draw.text((x, y), title, font=FONT_HEADER, fill=title_color)
    y += 30
    y = draw_wrapped_text(draw, (x, y), body, FONT_BODY, (30, 30, 30),
                          max_width, line_gap=6, paragraph_gap=8)
    return y + 16


def draw_numbered_box(draw, box, number, color, image_w, image_h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(float(x1), image_w - 1))
    y1 = max(0, min(float(y1), image_h - 1))
    x2 = max(0, min(float(x2), image_w - 1))
    y2 = max(0, min(float(y2), image_h - 1))

    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

    badge_text, badge_size = str(number), 28
    bx = int(x1)
    by = int(max(0, y1 - badge_size))
    if by < 2:
        by = int(y1 + 2)

    draw.rectangle([bx, by, bx + badge_size, by + badge_size],
                   fill=color, outline=color)

    try:
        bbox = draw.textbbox((0, 0), badge_text, font=FONT_BOX)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = 9, 13

    draw.text((bx + (badge_size - tw) / 2, by + (badge_size - th) / 2 - 1),
              badge_text, font=FONT_BOX, fill=(255, 255, 255))


# ============================================================
# 7. PANEL CONTENT
# ============================================================

def build_panel_title(label):
    label = normalize_label(label)
    if label == "entailment":
        return "VISIBLE EVIDENCE SUPPORTING HYPOTHESIS"
    if label == "contradiction":
        return "VISIBLE EVIDENCE CONTRADICTING HYPOTHESIS"
    return f"{label.upper()} VISUAL EVIDENCE"


def build_phrase_list_text(phrase_items, atom_to_ref):
    """The DINO query phrase and, when different, the shorter match phrase
    used later for Flickr30k Entities matching."""
    lines, seen = [], set()

    for idx, item in enumerate(phrase_items, start=1):
        phrase = clean_spacing(item.get("phrase", ""))
        match_phrase = clean_spacing(item.get("match_phrase", ""))
        if not phrase:
            continue

        ref = get_atom_ref(item, atom_to_ref)
        key = (ref, phrase.lower(), match_phrase.lower())
        if key in seen:
            continue
        seen.add(key)

        prefix = f"{ref}." if ref else f"{idx}."
        if (match_phrase and match_phrase.upper() != "NONE"
                and match_phrase.lower() != phrase.lower()):
            lines.append(f"{prefix} {phrase}  [match: {match_phrase}]")
        else:
            lines.append(f"{prefix} {phrase}")

    if not lines:
        return "No groundable visual evidence phrase was generated."
    return "\n".join(lines)


def build_reasoning_text(phrase_items, atom_to_ref, full_reason):
    lines, seen = [], set()

    for idx, item in enumerate(phrase_items, start=1):
        reason = clean_spacing(item.get("vlm_reasoning", ""))
        if not reason:
            continue
        if reason.lower() in seen:
            continue
        seen.add(reason.lower())

        ref = get_atom_ref(item, atom_to_ref)
        lines.append(f"{ref}. {reason}" if ref else f"{idx}. {reason}")

    return "\n".join(lines) if lines else clean_spacing(full_reason)


def build_boxes_text(drawn_detections):
    if not drawn_detections:
        return "No boxes were found above the threshold."
    return "\n".join(
        f"{idx}. {clean_spacing(det.get('label', '')) or 'object'}"
        for idx, det in enumerate(drawn_detections, start=1))


def draw_panel_content(draw, panel_x, panel_w, canvas_w, label, hypothesis,
                       atomic_facts, full_reason, phrase_items,
                       all_detections, drawn_detections):
    label = normalize_label(label)
    color = get_color(label)

    pad = 16
    inner_x = panel_x + pad
    inner_w = panel_w - 2 * pad
    header_h = 44

    draw.rectangle([panel_x, 0, canvas_w, header_h], fill=color)
    draw_centered_wrapped_text(draw, inner_x, canvas_w - pad, 10,
                               build_panel_title(label), FONT_TITLE,
                               (255, 255, 255), line_gap=2)

    y = header_h + 20

    display_atoms, atom_to_ref = build_atomic_fact_mapping(atomic_facts, phrase_items)

    if label == "contradiction":
        hypothesis_title = "Hypothesis claim"
        evidence_title = "Grounding phrase(s) from visible evidence"
    elif label == "entailment":
        hypothesis_title = "Hypothesis"
        evidence_title = "Grounding phrase(s) supporting hypothesis"
    else:
        hypothesis_title = "Hypothesis"
        evidence_title = "Grounding phrase(s)"

    y = draw_section(draw, inner_x, y, hypothesis_title,
                     shorten_text(hypothesis, 360), inner_w, color)

    y = draw_section(draw, inner_x, y, "Atomic facts",
                     shorten_multiline_text(build_atomic_facts_text(display_atoms), 700),
                     inner_w, color)

    y = draw_section(draw, inner_x, y, evidence_title,
                     shorten_multiline_text(
                         build_phrase_list_text(phrase_items, atom_to_ref), 650),
                     inner_w, color)

    reasoning_text = build_reasoning_text(phrase_items, atom_to_ref, full_reason)
    if reasoning_text:
        y = draw_section(draw, inner_x, y, "VLM reasoning",
                         shorten_multiline_text(reasoning_text, 900), inner_w, color)

    boxes_title = (f"Evidence bounding boxes: {len(drawn_detections)} shown "
                   f"from {len(all_detections)} detections")
    y = draw_section(draw, inner_x, y, boxes_title,
                     shorten_multiline_text(build_boxes_text(drawn_detections), 420),
                     inner_w, color)

    return y + 8


def draw_clean_grounding_figure(image, all_detections, drawn_detections,
                                phrase_items, label, hypothesis, atomic_facts,
                                full_reason, output_path):
    label = normalize_label(label)
    color = get_color(label)
    image_w, image_h = image.size

    panel_w = max(500, int(image_w * 0.98))
    canvas_w = image_w + panel_w

    # Lay the panel out once on a scratch canvas to find how tall it needs to be.
    temp_canvas = Image.new("RGB", (canvas_w, 2000), (255, 255, 255))
    panel_bottom = draw_panel_content(
        ImageDraw.Draw(temp_canvas), image_w, panel_w, canvas_w, label,
        hypothesis, atomic_facts, full_reason, phrase_items,
        all_detections, drawn_detections)

    canvas_h = max(image_h, panel_bottom + 8)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, image_w - 1, image_h - 1],
                   outline=(180, 180, 180), width=1)

    for idx, det in enumerate(drawn_detections, start=1):
        draw_numbered_box(draw, det["box"], idx, color, image_w, image_h)

    draw.rectangle([image_w, 0, canvas_w, canvas_h], fill=(250, 250, 250))

    draw_panel_content(draw, image_w, panel_w, canvas_w, label, hypothesis,
                       atomic_facts, full_reason, phrase_items,
                       all_detections, drawn_detections)

    canvas.save(output_path)


# ============================================================
# 8. REPORT-READY IMAGE
# ============================================================

def scale_box(box, scale_x, scale_y):
    x1, y1, x2, y2 = box
    return [float(x1) * scale_x, float(y1) * scale_y,
            float(x2) * scale_x, float(y2) * scale_y]


def draw_numbered_box_for_report(draw, box, number, color, image_w, image_h):
    """Thicker boxes and larger badges, so the image stays readable after the
    report resize."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(float(x1), image_w - 1))
    y1 = max(0, min(float(y1), image_h - 1))
    x2 = max(0, min(float(x2), image_w - 1))
    y2 = max(0, min(float(y2), image_h - 1))

    draw.rectangle([x1, y1, x2, y2], outline=color, width=REPORT_BOX_WIDTH)

    badge_text, badge_size = str(number), REPORT_BADGE_SIZE
    bx = int(x1)
    by = int(max(0, y1 - badge_size))
    if by < 2:
        by = int(y1 + 2)

    draw.rectangle([bx, by, bx + badge_size, by + badge_size],
                   fill=color, outline=color)

    try:
        font = load_font(max(16, int(badge_size * 0.52)), bold=True)
        bbox = draw.textbbox((0, 0), badge_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        font = FONT_BOX
        tw, th = 9, 13

    draw.text((bx + (badge_size - tw) / 2, by + (badge_size - th) / 2 - 1),
              badge_text, font=font, fill=(255, 255, 255))


def draw_report_grounding_image(image, drawn_detections, label, output_png_path):
    """The image and its boxes only. Hypothesis, labels and reasoning are left
    to LaTeX so the text stays sharp."""
    label = normalize_label(label)
    color = get_color(label)

    original_w, original_h = image.size
    if original_w <= 0 or original_h <= 0:
        raise ValueError("Invalid image size")

    scale = REPORT_TARGET_WIDTH / float(original_w)
    target_w = int(round(original_w * scale))
    target_h = int(round(original_h * scale))

    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS

    report_img = image.resize((target_w, target_h), resample_filter)
    draw = ImageDraw.Draw(report_img)

    # A thin border helps when the image sits in a multi-example LaTeX grid.
    draw.rectangle([0, 0, target_w - 1, target_h - 1],
                   outline=(180, 180, 180), width=REPORT_BORDER_WIDTH)

    for idx, det in enumerate(drawn_detections, start=1):
        draw_numbered_box_for_report(draw, scale_box(det["box"], scale, scale),
                                     idx, color, target_w, target_h)

    report_img.save(output_png_path)
    return output_png_path


# ============================================================
# 9. GROUNDING DINO
# ============================================================

def load_grounding_model(device):
    print(SEP)
    print("LOADING GROUNDING DINO")
    print(SEP)
    print(f"Model : {MODEL_ID}")
    print(f"Device: {device}")

    kwargs = {}
    if config.HF_CACHE_DIR:
        kwargs["cache_dir"] = config.HF_CACHE_DIR

    processor = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, dtype=torch.float32, **kwargs).to(device).eval()

    print("Grounding DINO loaded.\n")
    return processor, model


def run_grounding_dino_for_phrase(processor, model, device, image, phrase_item):
    phrase = safe_text(phrase_item.get("phrase", ""))
    text_query = make_text_query(phrase)

    inputs = processor(images=image, text=text_query, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]])[0]

    boxes = results["boxes"].detach().cpu()
    scores = results["scores"].detach().cpu()
    labels = results.get("text_labels", results.get("labels", []))

    detections = []
    for box, score, det_label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [round(float(v), 2) for v in box.tolist()]
        detections.append({
            "label": safe_text(det_label),
            "score": round(float(score.item()), 6),
            "box": [x1, y1, x2, y2],
            "source_phrase": phrase,
            "source_phrase_index": phrase_item.get("phrase_index", None),
            "source_atom": phrase_item.get("atom", ""),
            "source_atom_label": phrase_item.get("atom_label", ""),
            "source_vlm_reasoning": phrase_item.get("vlm_reasoning", ""),
            "text_query": text_query,
        })

    detections = sorted(detections, key=lambda x: x["score"], reverse=True)

    return {
        "phrase": phrase,
        "text_query": text_query,
        "grounding_found": len(detections) > 0,
        "num_boxes": len(detections),
        "best_detection": detections[0] if detections else None,
        "detections": detections,
    }


def run_grounding_for_all_phrases(processor, model, device, image, phrase_items):
    phrase_results, all_detections = [], []

    for phrase_item in phrase_items:
        phrase_result = run_grounding_dino_for_phrase(
            processor, model, device, image, phrase_item)
        phrase_results.append(phrase_result)
        all_detections.extend(phrase_result["detections"])

    all_detections = sorted(all_detections, key=lambda x: x["score"], reverse=True)
    unique_detections = select_unique_detections(all_detections)
    drawn_detections = unique_detections[:MAX_BOXES_TO_DRAW]

    return {
        "grounding_found": len(all_detections) > 0,
        "num_boxes": len(all_detections),
        "num_unique_boxes": len(unique_detections),
        "num_phrases_queried": len(phrase_items),
        "best_detection": all_detections[0] if all_detections else None,
        "best_unique_detection": unique_detections[0] if unique_detections else None,
        "all_detections": all_detections,
        "unique_detections": unique_detections,
        "drawn_detections": drawn_detections,
        "phrase_results": phrase_results,
        "max_boxes_drawn": MAX_BOXES_TO_DRAW,
        "display_box_rule": "top_unique_boxes_after_near_duplicate_suppression",
        "duplicate_suppression_used_for_display": True,
        "duplicate_coord_tolerance": DUPLICATE_COORD_TOLERANCE,
        "duplicate_iou_threshold": DUPLICATE_IOU_THRESHOLD,
        "duplicate_containment_threshold": DUPLICATE_CONTAINMENT_THRESHOLD,
        "duplicate_size_ratio_threshold": DUPLICATE_SIZE_RATIO_THRESHOLD,
    }


# ============================================================
# 10. MAIN LOOP
# ============================================================

def get_display_label(rec):
    """The figure explains the model's own decision, so the predicted label is
    what gets visualised, not the gold label."""
    return normalize_label(rec.get("final_label", rec.get("prediction", "")))


def copy_metadata_for_output(rec):
    keys = ["row_id", "row_key_occurrence", "Flickr30K_ID", "annotator_label",
            "gold", "hypothesis", "final_label", "prediction",
            "selected_candidate", "selected_model", "selected_method",
            "selected_prompt", "selected_candidate_label",
            "candidate_matches_learned_label"]
    return {k: rec.get(k, "") for k in keys if k in rec}


def skipped_record(rec, gold, final_label, hypothesis, atomic_facts,
                   full_reason, skip_reason):
    return {
        **copy_metadata_for_output(rec),
        "annotator_label": gold,
        "gold": gold,
        "final_label": final_label,
        "hypothesis": hypothesis,
        "atomic_facts": atomic_facts,
        "reason": full_reason,
        "grounding_phrases": rec.get("grounding_phrases", []),
        "grounding_phrase_summary": rec.get("grounding_phrase_summary", {}),
        "grounding_phrase": rec.get("grounding_phrase", {}),
        "grounding_dino": {
            "skipped": True,
            "skip_reason": skip_reason,
            "grounding_found": False,
            "num_boxes": 0,
            "num_phrases_queried": 0,
            "best_detection": None,
            "all_detections": [],
            "drawn_detections": [],
            "phrase_results": [],
            "max_boxes_drawn": MAX_BOXES_TO_DRAW,
        },
    }


def run(split, eval_name, limit, make_images):
    paths = build_paths(split, eval_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(SEP)
    print(f"GROUNDING DINO: {split.upper()}")
    print(SEP)
    print(f"Input     : {paths['input_jsonl']}")
    print(f"Output    : {paths['output_jsonl']}")
    print(f"Images    : {'on' if make_images else 'off'}\n")

    if not os.path.exists(paths["input_jsonl"]):
        raise FileNotFoundError(
            f"No phrase file at {paths['input_jsonl']}. "
            f"Run src.grounding.phrases first.")

    processor, model = load_grounding_model(device)

    seen = written = skipped = failed = no_box = 0
    final_label_counts = {"entailment": 0, "contradiction": 0,
                          "neutral": 0, "other": 0}
    gold_label_counts = dict(final_label_counts)

    with jsonlines.open(paths["input_jsonl"], "r") as reader, \
         jsonlines.open(paths["output_jsonl"], "w") as writer, \
         jsonlines.open(paths["failed_jsonl"], "w") as failed_writer:

        for rec in reader:
            if limit is not None and seen >= limit:
                break
            seen += 1

            try:
                flickr_id = safe_text(rec.get("Flickr30K_ID", ""))
                gold = normalize_label(rec.get("gold", rec.get("annotator_label", "")))
                final_label = get_display_label(rec)
                hypothesis = clean_spacing(rec.get("hypothesis", ""))
                atomic_facts = ensure_atomic_facts(rec.get("atomic_facts", []))
                full_reason = clean_spacing(rec.get("reason", ""))

                phrase_items = get_grounding_phrase_items(rec)

                # The phrase extractor already filters to entailment and
                # contradiction. This guard keeps the stage safe if a mixed
                # file is passed in.
                if final_label not in GROUNDABLE_FINAL_LABELS:
                    skipped += 1
                    writer.write(skipped_record(
                        rec, gold, final_label, hypothesis, atomic_facts,
                        full_reason, "final_label_not_entailment_or_contradiction"))
                    writer._fp.flush()
                    written += 1
                    continue

                if not phrase_items:
                    skipped += 1
                    writer.write(skipped_record(
                        rec, gold, final_label, hypothesis, atomic_facts,
                        full_reason, "no_groundable_phrases"))
                    writer._fp.flush()
                    written += 1
                    continue

                image, image_path = load_image(flickr_id)

                grounding_result = run_grounding_for_all_phrases(
                    processor, model, device, image, phrase_items)

                if not grounding_result["grounding_found"]:
                    no_box += 1

                final_key = final_label if final_label in final_label_counts else "other"
                gold_key = gold if gold in gold_label_counts else "other"
                final_label_counts[final_key] += 1
                gold_label_counts[gold_key] += 1

                phrase_part = "_".join(safe_filename(item["phrase"])[:30]
                                       for item in phrase_items[:2])
                row_id = safe_text(rec.get("row_id", seen))
                output_image_name = safe_filename(
                    f"row{row_id}_{flickr_id}_pred-{final_label}_gold-{gold}"
                    f"_{phrase_part}_ave_ls.png")

                output_image_path = os.path.join(
                    paths["image_output_dir"], output_image_name)
                report_image_path = os.path.join(
                    paths["report_image_output_dir"],
                    output_image_name.replace(".png", "_report.png"))

                if make_images:
                    draw_clean_grounding_figure(
                        image.copy(),
                        grounding_result["all_detections"],
                        grounding_result["drawn_detections"],
                        phrase_items, final_label, hypothesis,
                        atomic_facts, full_reason, output_image_path)

                    draw_report_grounding_image(
                        image.copy(), grounding_result["drawn_detections"],
                        final_label, report_image_path)

                writer.write({
                    **copy_metadata_for_output(rec),
                    "Flickr30K_ID": flickr_id,
                    "annotator_label": gold,
                    "gold": gold,
                    "final_label": final_label,
                    "prediction": normalize_label(rec.get("prediction", final_label)),
                    "hypothesis": hypothesis,
                    "atomic_facts": atomic_facts,
                    "reason": full_reason,
                    "image_path": image_path,
                    "image_width": image.width,
                    "image_height": image.height,
                    "annotated_image_path": output_image_path if make_images else "",
                    "report_image_path": report_image_path if make_images else "",
                    "grounding_phrases": rec.get("grounding_phrases", []),
                    "grounding_phrase_summary": rec.get("grounding_phrase_summary", {}),
                    "grounding_phrase": rec.get("grounding_phrase", {}),
                    "grounding_dino": {
                        "skipped": False,
                        "model": MODEL_ID,
                        "box_threshold": BOX_THRESHOLD,
                        "text_threshold": TEXT_THRESHOLD,
                        "visualized_label": final_label,
                        "gold_label": gold,
                        "is_false_entailment_or_contradiction": bool(gold != final_label),
                        **grounding_result,
                    },
                })
                writer._fp.flush()
                written += 1

                if written % PROGRESS_EVERY == 0:
                    print(f"Written: {written} | Seen: {seen} | Skipped: {skipped} "
                          f"| No box: {no_box} | Failed: {failed}", flush=True)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                failed += 1
                print(f"  ERROR at row {rec.get('row_id', seen)}: {e}", flush=True)
                failed_writer.write({
                    "row_id": rec.get("row_id", None),
                    "Flickr30K_ID": safe_text(rec.get("Flickr30K_ID", "")),
                    "hypothesis": safe_text(rec.get("hypothesis", "")),
                    "error": str(e),
                    "traceback": traceback.format_exc()[:4000],
                })
                failed_writer._fp.flush()
                continue

    print("")
    print(SEP)
    print("GROUNDING SUMMARY")
    print(SEP)
    print(f"Seen    : {seen}")
    print(f"Written : {written}")
    print(f"Skipped : {skipped}")
    print(f"No box  : {no_box}")
    print(f"Failed  : {failed}")
    print(f"Final label counts: {final_label_counts}")
    print(f"Gold label counts : {gold_label_counts}")
    print(f"\nWritten to {paths['output_jsonl']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--eval-name", default="AVE_learned_selection_evaluation_v3")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-images", action="store_true",
                    help="skip figure rendering, keeping only the detection records")
    args = ap.parse_args()
    run(args.split, args.eval_name, args.limit, make_images=not args.no_images)


if __name__ == "__main__":
    main()