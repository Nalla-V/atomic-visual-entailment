import argparse
import os

import jsonlines
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

# ============================================================
# 1. CONFIGURATION
# ============================================================
MODEL_ID = "microsoft/Florence-2-large"
BASE_DIR = os.environ.get("DATA_ROOT", ".")
IMAGE_DIR = os.path.join(BASE_DIR, "Input/flickr30k_images")
OUTPUT_FILE = os.path.join(BASE_DIR, "Output/generated_captions_v2.jsonl")

HF_CACHE_DIR = os.environ.get("HF_HOME") or None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ============================================================
# 2. LOAD MODEL & PROCESSOR
# ============================================================
print(f"Loading Florence-2 from the HuggingFace Hub: {MODEL_ID}")

_kwargs = {"trust_remote_code": True}
if HF_CACHE_DIR:
    _kwargs["cache_dir"] = HF_CACHE_DIR

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    **_kwargs
).to(DEVICE).eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, **_kwargs)

# ============================================================
# 3. RESUME LOGIC
# ============================================================
processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    try:
        with jsonlines.open(OUTPUT_FILE, mode='r') as reader:
            for obj in reader:
                processed_ids.add(str(obj.get("Flickr30K_ID")))
    except Exception:
        pass
print(f"Resuming: {len(processed_ids)} already processed.")

# ============================================================
# 4. MAIN PROCESSING LOOP
# ============================================================
def main(limit=None):
    # Gather all images and filter out what is already done
    all_images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')])
    images_to_process = [f for f in all_images if f.replace('.jpg', '') not in processed_ids]

    if limit:
        images_to_process = images_to_process[:limit]

    total_to_do = len(images_to_process)
    print(f"Total: {len(all_images)} | Remaining: {total_to_do}")

    if total_to_do == 0:
        print("Everything is already processed!")
        return

    # Append mode ('a') ensures we don't lose data if the job times out
    with jsonlines.open(OUTPUT_FILE, mode='a') as writer:
        for i, filename in enumerate(tqdm(images_to_process, desc="Captioning", mininterval=60)):
            img_id = filename.replace('.jpg', '')
            img_path = os.path.join(IMAGE_DIR, filename)

            try:
                # Load and Convert
                image = Image.open(img_path).convert("RGB")

                # We use <DETAILED_CAPTION> for rich NLI context
                prompt = "<DETAILED_CAPTION>" 

                # Preprocess
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
                inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

                # Inference
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512,
                        num_beams=5,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )

                # Post-process
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                parsed_answer = processor.post_process_generation(
                    generated_text,
                    task=prompt,
                    image_size=(image.width, image.height)
                )

                # Save Result
                writer.write({
                    "Flickr30K_ID": img_id,
                    "generated_caption": parsed_answer[prompt]
                })

                # Status log for Slurm
                if (i + 1) % 1000 == 0:
                    print(f"[PROGRESS] {i + 1}/{total_to_do} images completed.")

            except Exception as e:
                print(f"[ERROR] Failed ID {img_id}: {e}")
                continue


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Florence-2 captions.")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N images, for testing")
    args = ap.parse_args()
    main(limit=args.limit)