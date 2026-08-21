"""VLM prediction over hypotheses and atomic facts.

    python -m src.prediction.predict --vlm internvl --split dev --method joint

Both prompt variants run from one model load. Resume is by input line index,
so identical rows are predicted independently.

Passing --image black or --image white replaces the real image with a blank
one, which is the hypothesis-bias check. That restricts the run to the
full-hypothesis and joint atomic methods and writes to a separate folder.
"""

import argparse
import os
import sys
import traceback

import jsonlines

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src import config
from src.prediction import models
from src.prediction.common import (
    safe_text,
    normalize_label,
    ensure_list_of_atoms,
    count_lines,
    ensure_parent_dir,
)

METHOD_MODULES = {}


def _load_method(name):
    if name not in METHOD_MODULES:
        METHOD_MODULES[name] = __import__(
            f"src.prediction.{name}", fromlist=[name]
        )
    return METHOD_MODULES[name]


def read_input_record(obj, vlm):
    return {
        "vlm": vlm,
        "img_id": safe_text(obj.get("Flickr30K_ID", "")),
        "hypothesis": safe_text(obj.get("hypothesis", "")),
        "gold": normalize_label(obj.get("annotator_label", "")),
        "atoms": ensure_list_of_atoms(
            obj.get("atomic_facts") or obj.get("raw_atoms") or []
        ),
    }


def _error_result(message):
    """A record is still written on failure, so output rows match input rows."""
    scores = {label: 1.0 / len(config.FINAL_LABELS) for label in config.FINAL_LABELS}
    return {
        "strategy": "error",
        "prompt_style": "",
        "scores": scores,
        "prediction": "",
        "model_prediction": "",
        "score_prediction": "",
        "aggregation_mismatch": False,
        "reason": f"Error: {message}",
        "atom_observations": [],
        "decomposed_atoms": [],
        "per_atom": [],
        "parse_ok": False,
        "parse_error": message,
        "fallback_used": False,
        "parse_mode": "",
        "parser": "plain",
        "hf_id": "",
        "vlm_key": "",
        "confidence_score": 0.0,
        "margin": 0.0,
        "entropy": 0.0,
        "normalized_entropy": 0.0,
        "top_label": "",
        "second_label": "",
    }


def output_paths(vlm, vlm_dir, method, variant, split, image):
    """Real-image runs go to the normal prediction folder. Blank-image runs go
    to the hypothesis-bias folder for that VLM and colour."""
    if image == "real":
        main_path = config.prediction_file(vlm_dir, method.OUTPUT_NAME, variant, split)
        debug_path = config.prediction_file(
            vlm_dir, method.OUTPUT_NAME, variant, split, debug=True)
    else:
        bias_dir = models.BIAS_DIRS[vlm]
        main_path = config.bias_prediction_file(
            bias_dir, image, method.OUTPUT_NAME, variant, split)
        debug_path = config.bias_prediction_file(
            bias_dir, image, method.OUTPUT_NAME, variant, split, debug=True)
    return main_path, debug_path


def run_method(adapter, vlm, method_name, input_file, vlm_dir,
               split, limit, progress_every, image="real"):
    method = _load_method(method_name)
    variants = list(method.VARIANTS)

    paths = {}
    for variant in variants:
        main_path, debug_path = output_paths(
            vlm, vlm_dir, method, variant, split, image)
        ensure_parent_dir(main_path)
        paths[variant] = (main_path, debug_path)

    # Resume by line count, taking the minimum so an interrupted record is redone.
    done = {v: count_lines(paths[v][0]) for v in variants}
    start_at = min(done.values())

    print(f"\n=== {method_name} ===", flush=True)
    for v in variants:
        print(f"  {v:<12} {done[v]:>7} rows -> {paths[v][0]}")
    print(f"  resuming at input row {start_at}", flush=True)

    blank = None if image == "real" else image

    writers = {}
    try:
        for v in variants:
            writers[v] = (
                jsonlines.open(paths[v][0], "a"),
                jsonlines.open(paths[v][1], "a"),
            )

        processed = 0
        failed = 0

        with jsonlines.open(input_file, "r") as reader:
            for i, obj in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                if i < start_at:
                    continue

                record = method.prepare(read_input_record(obj, vlm))

                try:
                    image_ref = adapter.load_image(record["img_id"], blank=blank)

                    for v in variants:
                        if i < done[v]:
                            continue
                        result = method.score(adapter, image_ref, record, v)
                        _write(writers[v], method, record, result, i)

                    processed += 1
                    if processed % progress_every == 0:
                        print(f"  row {i + 1}, processed {processed}", flush=True)

                except Exception as e:
                    failed += 1
                    print(f"  ERROR at row {i} ({record['img_id']}): {e}", flush=True)
                    for v in variants:
                        if i < done[v]:
                            continue
                        _write(writers[v], method, record,
                               _error_result(str(e)), i, error=True)
                    continue

        print(f"  done: processed={processed} failed={failed}", flush=True)

    finally:
        for v in writers:
            for w in writers[v]:
                w.close()


def _write(writer_pair, method, record, result, row_index, error=False):
    main_writer, debug_writer = writer_pair

    main_writer.write(method.build_record(record, result))
    main_writer._fp.flush()
    os.fsync(main_writer._fp.fileno())

    debug_entry = {
        "row_index": row_index,
        "Flickr30K_ID": record["img_id"],
        "hypothesis": record["hypothesis"],
        "atoms_fallback": record.get("atoms_fallback", False),
        "parse_ok": result.get("parse_ok", False),
        "parse_error": result.get("parse_error", ""),
    }
    if error:
        debug_entry["traceback"] = traceback.format_exc()[:4000]
    else:
        debug_entry["raw_response"] = result.get("raw_response", "")

    debug_writer.write(debug_entry)
    debug_writer._fp.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vlm", required=True, choices=sorted(models.VLM_MODELS))
    ap.add_argument("--split", required=True, choices=config.SPLITS)
    ap.add_argument("--method", required=True,
                    help="one method, or 'all' for every method this VLM supports")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--input-file", default=None)
    ap.add_argument("--decomposer", default="qwen32")
    ap.add_argument("--image", default="real", choices=["real", "black", "white"],
                    help="blank image for the hypothesis-bias check")
    args = ap.parse_args()

    available = models.methods_for(args.vlm)

    # The bias check covers full-hypothesis and joint atomic prediction only.
    if args.image != "real":
        if args.vlm not in models.BIAS_DIRS:
            raise SystemExit(
                f"The hypothesis-bias check covers: "
                f"{', '.join(sorted(models.BIAS_DIRS))}")
        available = [m for m in available if m in models.BIAS_METHODS]

    if args.method == "all":
        methods = available
    elif args.method in available:
        methods = [args.method]
    else:
        raise SystemExit(f"{args.vlm} supports: {', '.join(available)}, or 'all'")

    input_file = args.input_file or config.atoms_file(args.split, args.decomposer)
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"No input at {input_file}. Run decomposition first, or check DATA_ROOT "
            f"(currently {config.DATA_ROOT})."
        )

    vlm_dir = models.OUTPUT_DIRS[args.vlm]
    progress_every = config.PROGRESS_EVERY[args.split]

    print(f"VLM     : {args.vlm}")
    print(f"Split   : {args.split}")
    print(f"Methods : {', '.join(methods)}")
    print(f"Image   : {args.image}")
    print(f"Input   : {input_file}", flush=True)

    adapter = models.load(args.vlm)

    for method_name in methods:
        run_method(adapter, args.vlm, method_name, input_file, vlm_dir,
                   args.split, args.limit, progress_every, args.image)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()