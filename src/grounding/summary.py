"""Evaluate grounding against Flickr30k Entities annotations.

    python -m src.grounding.summary --split dev

Matches each grounding query to human-annotated entity phrases, then reports
Recall@1, Recall@3 and mean best IoU@3 over the matched rows.
"""

import argparse
import csv
import json
import os
import re
import string
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import config

EVAL_SUBDIR = "AVE_learned_selection_evaluation_v3"
GROUNDING_SUBDIR = "grounding_dino_v1"

IOU_THRESHOLD = 0.50

# A query matches an entity phrase when one contains the other, or when token
# overlap and Jaccard both clear these floors.
MIN_OVERLAP = 0.60
MIN_JACCARD = 0.20
MAX_MATCHED_ENTITIES = 8

DEDUP_IOU = 0.92
DEDUP_COORD_TOL = 3.0

TARGET_PREDICTED_LABELS = ["entailment", "contradiction"]

STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "in", "on", "at", "by", "with", "without", "near", "next", "to",
    "of", "for", "from", "into", "onto", "over", "under", "while",
    "and", "or", "as", "is", "are", "was", "were", "be", "being",
    "his", "her", "their", "its", "it", "he", "she", "they", "them",
}

PLURAL_NORMALIZATION = {
    "children": "child",
    "kids": "kid",
    "men": "man",
    "women": "woman",
    "people": "person",
    "boys": "boy",
    "girls": "girl",
}

# Flickr30k sentence tags look like [EN#12345/people young boys]
ENTITY_TAG_RE = re.compile(r"\[(?:/)?EN#(\d+)(?:/([^\s\]]+))?\s+([^\]]+)\]")


# ============================================================
# PATHS
# ============================================================

def grounding_jsonl_path(split, eval_name):
    return os.path.join(config.OUTPUT_DIR, f"{split}_dataset", eval_name,
                        GROUNDING_SUBDIR,
                        f"grounding_dino_boxes_ave_ls_final_{split}.jsonl")


def output_dir(split, eval_name):
    return os.path.join(config.OUTPUT_DIR, f"{split}_dataset", eval_name,
                        GROUNDING_SUBDIR)


def annotations_zip_path():
    return os.path.join(config.INPUT_DIR, "flickr30k_entities", "annotations.zip")


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, list):
        return " ".join(safe_text(v) for v in x if safe_text(v))
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()


def clean_spacing(text):
    text = safe_text(text)
    text = re.sub(r"[ \t\r\n]+", " ", text).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", text)


def get_first(rec, keys, default=""):
    for key in keys:
        if key in rec and rec[key] not in (None, ""):
            return rec[key]
    return default


def normalize_label(x):
    text = safe_text(x).lower().strip()
    if text in set(config.FINAL_LABELS):
        return text
    if text in {"e", "entails", "entailed"}:
        return "entailment"
    if text in {"n", "neutrality"}:
        return "neutral"
    if text in {"c", "contradict", "contradicted"}:
        return "contradiction"
    return text


def label_display(label):
    label = normalize_label(label)
    if label in {"entailment", "contradiction", "neutral"}:
        return label[:1].upper() + label[1:]
    if not label:
        return "Unknown"
    return label[:1].upper() + label[1:]


def image_id_key(x):
    text = os.path.basename(safe_text(x))
    return re.sub(r"\.(jpg|jpeg|png|xml|txt)$", "", text, flags=re.IGNORECASE).strip()


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def round_metric(x):
    return round(float(x), 4)


# ============================================================
# BOX MATHS
# ============================================================

def valid_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def box_area(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_intersection(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_iou(box_a, box_b):
    inter = box_intersection(box_a, box_b)
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def boxes_near_identical(box_a, box_b, coord_tol=DEDUP_COORD_TOL):
    return all(abs(float(a) - float(b)) <= coord_tol for a, b in zip(box_a, box_b))


def deduplicate_detections(detections):
    valid = [d for d in detections if valid_box(d.get("box"))]
    valid = sorted(valid, key=lambda d: float(d.get("score", 0.0)), reverse=True)

    unique = []
    for det in valid:
        box = det["box"]
        duplicate = any(boxes_near_identical(box, chosen["box"])
                        or box_iou(box, chosen["box"]) >= DEDUP_IOU
                        for chosen in unique)
        if not duplicate:
            unique.append(det)
    return unique


def best_iou(pred_boxes, gt_boxes):
    if not pred_boxes or not gt_boxes:
        return 0.0
    return max(box_iou(p, g) for p in pred_boxes for g in gt_boxes)


# ============================================================
# FLICKR30K ENTITIES READER
# ============================================================

def parse_flickr_xml_bytes(xml_bytes):
    boxes_by_entity = defaultdict(list)
    image_size = {"width": 0, "height": 0}

    root = ET.fromstring(xml_bytes)

    size_node = root.find("size")
    if size_node is not None:
        try:
            image_size["width"] = int(float(size_node.findtext("width", "0")))
            image_size["height"] = int(float(size_node.findtext("height", "0")))
        except Exception:
            pass

    for obj in root.findall("object"):
        names = [safe_text(n.text) for n in obj.findall("name") if safe_text(n.text)]
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            box = [float(bnd.findtext("xmin", "0")), float(bnd.findtext("ymin", "0")),
                   float(bnd.findtext("xmax", "0")), float(bnd.findtext("ymax", "0"))]
        except Exception:
            continue
        if not valid_box(box):
            continue
        for name in names:
            boxes_by_entity[name].append(box)

    return dict(boxes_by_entity), image_size


def parse_flickr_sentence_bytes(txt_bytes):
    phrases_by_entity = defaultdict(list)
    text = txt_bytes.decode("utf-8", errors="ignore")

    for line in text.splitlines():
        for match in ENTITY_TAG_RE.finditer(line):
            phrase = clean_spacing(match.group(3))
            if phrase:
                phrases_by_entity[match.group(1)].append({
                    "phrase": phrase,
                    "category": safe_text(match.group(2)),
                })

    return dict(phrases_by_entity)


class Flickr30kZipReader:
    """Reads the Entities annotations directly from annotations.zip."""

    def __init__(self, zip_path):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Annotation zip not found: {zip_path}")
        self.zip_path = zip_path
        self.zf = zipfile.ZipFile(zip_path, "r")
        self.xml_members, self.txt_members = {}, {}
        self._build_index()

    def close(self):
        self.zf.close()

    def _build_index(self):
        for info in self.zf.infolist():
            name = info.filename.replace("\\", "/")
            img_id = image_id_key(os.path.basename(name))
            if name.lower().endswith(".xml"):
                self.xml_members[img_id] = name
            elif name.lower().endswith(".txt"):
                self.txt_members[img_id] = name

        if not self.xml_members:
            raise FileNotFoundError("No XML files found inside annotations.zip")
        if not self.txt_members:
            raise FileNotFoundError("No TXT files found inside annotations.zip")

    def load_image_entities(self, image_id):
        img_id = image_id_key(image_id)
        result = {
            "image_id": img_id,
            "xml_exists": img_id in self.xml_members,
            "txt_exists": img_id in self.txt_members,
            "entities": [],
            "image_size": {"width": 0, "height": 0},
        }

        if img_id not in self.xml_members or img_id not in self.txt_members:
            return result

        boxes_by_entity, image_size = parse_flickr_xml_bytes(
            self.zf.read(self.xml_members[img_id]))
        phrases_by_entity = parse_flickr_sentence_bytes(
            self.zf.read(self.txt_members[img_id]))
        result["image_size"] = image_size

        entities = []
        for entity_id in sorted(set(boxes_by_entity) | set(phrases_by_entity)):
            boxes = boxes_by_entity.get(entity_id, [])
            if not boxes:
                continue
            for phrase_item in phrases_by_entity.get(entity_id, []):
                entities.append({
                    "entity_id": entity_id,
                    "phrase": phrase_item["phrase"],
                    "category": phrase_item.get("category", ""),
                    "boxes": boxes,
                })

        result["entities"] = entities
        return result


# ============================================================
# PHRASE MATCHING
# ============================================================

def normalize_for_match(text):
    text = clean_spacing(text).lower().replace("'", "")
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    return re.sub(r"\s+", " ", text).strip()


def token_normalize(token):
    token = token.lower().strip()
    if token in PLURAL_NORMALIZATION:
        return PLURAL_NORMALIZATION[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def phrase_tokens(text):
    toks = []
    for tok in normalize_for_match(text).split():
        tok = token_normalize(tok)
        if tok and tok not in STOPWORDS:
            toks.append(tok)
    return toks


def phrase_match_score(query, candidate):
    """Returns score, overlap coefficient, Jaccard, contains-match flag."""
    q_norm, c_norm = normalize_for_match(query), normalize_for_match(candidate)
    if not q_norm or not c_norm:
        return 0.0, 0.0, 0.0, False

    contains_match = q_norm in c_norm or c_norm in q_norm

    q_tokens, c_tokens = set(phrase_tokens(query)), set(phrase_tokens(candidate))
    if not q_tokens or not c_tokens:
        return (1.0 if contains_match else 0.0), 0.0, 0.0, contains_match

    inter = q_tokens & c_tokens
    overlap = len(inter) / max(1, min(len(q_tokens), len(c_tokens)))
    jaccard = len(inter) / max(1, len(q_tokens | c_tokens))
    score = max(1.0 if contains_match else 0.0, 0.70 * overlap + 0.30 * jaccard)

    return score, overlap, jaccard, contains_match


def extract_grounding_queries(rec):
    """Prefer match_phrase, falling back to the grounding phrase itself."""
    queries = []

    def add_query(x):
        text = clean_spacing(x)
        if not text or text.upper() == "NONE":
            return
        if text.lower() not in {q.lower() for q in queries}:
            queries.append(text)

    items = rec.get("grounding_phrases", [])
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict) or item.get("can_ground", True) is False:
                continue
            match_phrase = clean_spacing(item.get("match_phrase", ""))
            phrase = clean_spacing(item.get("phrase", ""))
            add_query(match_phrase
                      if match_phrase and match_phrase.upper() != "NONE" else phrase)

    single = rec.get("grounding_phrase", {})
    if isinstance(single, dict) and single.get("can_ground", True) is not False:
        match_phrase = clean_spacing(single.get("match_phrase", ""))
        phrase = clean_spacing(single.get("phrase", ""))
        add_query(match_phrase
                  if match_phrase and match_phrase.upper() != "NONE" else phrase)

    if not queries:
        gd = rec.get("grounding_dino", {})
        if isinstance(gd, dict):
            for det in gd.get("all_detections", []) or []:
                if isinstance(det, dict):
                    add_query(det.get("source_phrase", ""))

    return queries


def match_entities_for_queries(queries, entities):
    candidates, seen = [], set()

    for query in queries:
        for ent in entities:
            phrase = clean_spacing(ent.get("phrase", ""))
            boxes = ent.get("boxes", []) or []
            if not phrase or not boxes:
                continue

            score, overlap, jaccard, contains_match = phrase_match_score(query, phrase)
            if not (contains_match
                    or (overlap >= MIN_OVERLAP and jaccard >= MIN_JACCARD)):
                continue

            key = (ent.get("entity_id", ""), phrase.lower())
            if key in seen:
                continue
            seen.add(key)

            candidates.append({
                "query": query,
                "entity_id": ent.get("entity_id", ""),
                "phrase": phrase,
                "category": ent.get("category", ""),
                "boxes": boxes,
                "score": round(score, 6),
                "overlap": round(overlap, 6),
                "jaccard": round(jaccard, 6),
                "contains_match": contains_match,
            })

    candidates.sort(key=lambda x: (x["score"], x["overlap"], x["jaccard"]),
                    reverse=True)
    return candidates[:MAX_MATCHED_ENTITIES]


# ============================================================
# DETECTIONS
# ============================================================

def get_detections(rec):
    gd = rec.get("grounding_dino", {})
    if not isinstance(gd, dict):
        return []
    for key in ["unique_detections", "all_detections", "drawn_detections"]:
        dets = gd.get(key, None)
        if isinstance(dets, list) and dets:
            return [d for d in dets if isinstance(d, dict)]
    return []


def flatten_gt_boxes(matches):
    boxes, seen = [], set()
    for match in matches:
        for box in match.get("boxes", []) or []:
            if not valid_box(box):
                continue
            key = tuple(round(float(v), 3) for v in box)
            if key in seen:
                continue
            seen.add(key)
            boxes.append([float(v) for v in box])
    return boxes


# ============================================================
# METRICS
# ============================================================

def safe_mean(values):
    return sum(values) / len(values) if values else 0.0


def safe_rate(num, den):
    return float(num) / den if den else 0.0


def summarize(rows):
    sampled = len(rows)
    matched_rows = [r for r in rows if int(r["annotation_matched"]) == 1]

    return {
        "sampled": sampled,
        "has_grounding_query": sum(int(r["has_query"]) for r in rows),
        "annotation_files_found": sum(int(r["annotation_files_found"]) for r in rows),
        "matched": len(matched_rows),
        "coverage": round_metric(safe_rate(len(matched_rows), sampled)),
        "grounding_box_rate": round_metric(
            safe_rate(sum(int(r["grounding_found"]) for r in rows), sampled)),
        "recall_at_1_iou_ge_0_50": round_metric(safe_rate(
            sum(int(r["hit_at_1"]) for r in matched_rows), len(matched_rows))),
        "recall_at_3_iou_ge_0_50": round_metric(safe_rate(
            sum(int(r["hit_at_3"]) for r in matched_rows), len(matched_rows))),
        "mean_best_iou_at_3": round_metric(safe_mean(
            [float(r["best_iou_at_3"]) for r in matched_rows])),
    }


def compact_summary_rows(metric_rows):
    """Entailment, Contradiction, Overall: the reported grounding table."""
    output_rows = []

    def make_row(condition, stats):
        return {
            "Condition": condition,
            "Sampled": stats["sampled"],
            "Matched": stats["matched"],
            "Coverage": stats["coverage"],
            "Recall@1": stats["recall_at_1_iou_ge_0_50"],
            "Recall@3": stats["recall_at_3_iou_ge_0_50"],
            "Mean best IoU@3": stats["mean_best_iou_at_3"],
        }

    for label in TARGET_PREDICTED_LABELS:
        group = [r for r in metric_rows
                 if normalize_label(r.get("final_label", "")) == label]
        output_rows.append(make_row(label_display(label), summarize(group)))

    output_rows.append(make_row("Overall", summarize(metric_rows)))
    return output_rows


def summarize_by_predicted_label(metric_rows):
    groups = defaultdict(list)
    for row in metric_rows:
        groups[normalize_label(row.get("final_label", ""))].append(row)
    return {label: summarize(rows) for label, rows in sorted(groups.items())}


# ============================================================
# MAIN
# ============================================================

def run(split, eval_name):
    grounding_jsonl = grounding_jsonl_path(split, eval_name)
    out_dir = output_dir(split, eval_name)
    os.makedirs(out_dir, exist_ok=True)

    annotations_zip = annotations_zip_path()

    metrics_json = os.path.join(out_dir, f"grounding_metrics_ave_ls_final_{split}.json")
    summary_csv = os.path.join(
        out_dir, f"grounding_metrics_summary_ave_ls_final_{split}.csv")
    row_csv = os.path.join(out_dir, f"grounding_metrics_rows_ave_ls_final_{split}.csv")

    if not os.path.exists(grounding_jsonl):
        raise FileNotFoundError(
            f"Grounding JSONL not found: {grounding_jsonl}. "
            f"Run src.grounding.detect first.")
    if not os.path.exists(annotations_zip):
        raise FileNotFoundError(f"annotations.zip not found: {annotations_zip}")

    print("Input grounding JSONL:", grounding_jsonl)
    print("Input annotations zip:", annotations_zip)

    zip_reader = Flickr30kZipReader(annotations_zip)
    entity_cache, metric_rows = {}, []

    try:
        for idx, rec in enumerate(read_jsonl(grounding_jsonl), start=1):
            flickr_id = image_id_key(get_first(
                rec, ["Flickr30K_ID", "flickr30k_id", "image_id", "image"], ""))
            row_id = safe_text(get_first(rec, ["row_id", "index", "id"], idx))
            gold = normalize_label(get_first(
                rec, ["gold", "gold_label", "annotator_label", "label"], ""))
            final_label = normalize_label(get_first(
                rec, ["final_label", "prediction", "predicted_label",
                      "final_prediction"], ""))
            hypothesis = clean_spacing(get_first(
                rec, ["hypothesis", "sentence2", "caption"], ""))

            queries = extract_grounding_queries(rec)

            if flickr_id not in entity_cache:
                entity_cache[flickr_id] = zip_reader.load_image_entities(flickr_id)
            ent_data = entity_cache[flickr_id]

            matched_entities = match_entities_for_queries(
                queries, ent_data.get("entities", []))
            gt_boxes = flatten_gt_boxes(matched_entities)

            raw_detections = get_detections(rec)
            unique_detections = deduplicate_detections(raw_detections)
            pred_boxes = [d["box"] for d in unique_detections if valid_box(d.get("box"))]

            top1_iou = best_iou(pred_boxes[:1], gt_boxes)
            top3_iou = best_iou(pred_boxes[:3], gt_boxes)
            annotation_matched = bool(gt_boxes)

            metric_rows.append({
                "row_index": idx,
                "row_id": row_id,
                "Flickr30K_ID": flickr_id,
                "gold": gold,
                "final_label": final_label,
                "condition": label_display(final_label),
                "prediction_correct": int(gold == final_label),
                "hypothesis": hypothesis,
                "grounding_queries": " | ".join(queries),
                "has_query": int(bool(queries)),
                "annotation_files_found": int(bool(
                    ent_data.get("xml_exists") and ent_data.get("txt_exists"))),
                "annotation_matched": int(annotation_matched),
                "num_matched_entities": len(matched_entities),
                "num_gt_boxes": len(gt_boxes),
                "matched_entity_phrases": " | ".join(
                    f"{m['entity_id']}:{m['phrase']}" for m in matched_entities[:5]),
                "grounding_found": int(bool(pred_boxes)),
                "num_raw_pred_boxes": len(raw_detections),
                "num_unique_pred_boxes": len(pred_boxes),
                "best_iou_at_1": round_metric(top1_iou),
                "best_iou_at_3": round_metric(top3_iou),
                "hit_at_1": int(annotation_matched and top1_iou >= IOU_THRESHOLD),
                "hit_at_3": int(annotation_matched and top3_iou >= IOU_THRESHOLD),
            })

            if idx % 500 == 0:
                print(f"Processed {idx} rows...", flush=True)

    finally:
        zip_reader.close()

    compact_rows = compact_summary_rows(metric_rows)

    summary = {
        "config": {
            "split": split,
            "grounding_jsonl": grounding_jsonl,
            "annotations_zip": annotations_zip,
            "iou_threshold": IOU_THRESHOLD,
            "phrase_matching_min_overlap": MIN_OVERLAP,
            "phrase_matching_min_jaccard": MIN_JACCARD,
            "max_matched_entities": MAX_MATCHED_ENTITIES,
            "dedup_iou": DEDUP_IOU,
            "dedup_coord_tolerance": DEDUP_COORD_TOL,
            "condition_definition":
                "Condition is based on the final predicted AVE-LS label.",
        },
        "overall": summarize(metric_rows),
        "by_predicted_label": summarize_by_predicted_label(metric_rows),
        "compact_table": compact_rows,
        "notes": {
            "coverage": (
                "Coverage is Matched divided by Sampled. A row is Matched when the "
                "extracted grounding query can be linked to at least one Flickr30k "
                "Entities human box."),
            "recall_denominator": (
                "Recall@1, Recall@3, and Mean best IoU@3 are computed only over "
                "Matched rows. Rows without a matched human box are excluded."),
            "recall_at_1": (
                "Recall@1 counts a matched row as correct if the top predicted box "
                "has IoU >= 0.50 with any matched human box."),
            "recall_at_3": (
                "Recall@3 counts a matched row as correct if any of the top three "
                "predicted boxes has IoU >= 0.50 with any matched human box."),
            "mean_best_iou_at_3": (
                "Mean best IoU@3 averages the best IoU obtained by any of the top "
                "three predicted boxes for each matched row."),
        },
    }

    write_json(metrics_json, summary)
    write_csv(summary_csv, compact_rows)
    write_csv(row_csv, metric_rows)

    print("\n================ Compact grounding metrics ================")
    print(f"Split: {split}")
    print(f"{'Condition':<15} {'Sampled':>8} {'Matched':>8} {'Coverage':>10} "
          f"{'Recall@1':>10} {'Recall@3':>10} {'MeanIoU@3':>12}")
    print("-" * 88)
    for row in compact_rows:
        print(f"{row['Condition']:<15} {int(row['Sampled']):>8} "
              f"{int(row['Matched']):>8} {float(row['Coverage']):>10.4f} "
              f"{float(row['Recall@1']):>10.4f} {float(row['Recall@3']):>10.4f} "
              f"{float(row['Mean best IoU@3']):>12.4f}")
    print("===========================================================\n")

    print("Saved:")
    print("  ", metrics_json)
    print("  ", summary_csv)
    print("  ", row_csv)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--eval-name", default=EVAL_SUBDIR)
    args = ap.parse_args()
    run(args.split, args.eval_name)


if __name__ == "__main__":
    main()