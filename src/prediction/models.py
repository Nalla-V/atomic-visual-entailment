"""VLM registry and adapters.

Families differ in three ways: which class loads them, whether the processor
takes a PIL image or a file path via qwen_vl_utils, and which precision they
run in. All are declared per model in VLM_MODELS.
"""

import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from src import config


VLM_MODELS = {
    "qwen3": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "loader": "qwen3vl",
        "images": "vision_info",
        "dtype": "bfloat16",
        "parser": "plain",
        "role": "main",
    },
    "internvl": {
        "hf_id": "OpenGVLab/InternVL3-8B-hf",
        "loader": "auto",
        "images": "pil",
        "dtype": "bfloat16",
        "parser": "plain",
        "role": "main",
    },
    "qwen3vl_32b": {
        "hf_id": "Qwen/Qwen3-VL-32B-Instruct",
        "loader": "qwen3vl",
        "images": "pil",
        "dtype": "bfloat16",
        "parser": "recovery",
        "role": "comparison",
    },
    "qwen2vl_2b": {
        "hf_id": "Qwen/Qwen2-VL-2B-Instruct",
        "loader": "qwen2vl",
        "images": "pil",
        "dtype": "bfloat16",
        "parser": "recovery",
        "role": "comparison",
    },
    "internvl3_1b": {
        "hf_id": "OpenGVLab/InternVL3-1B-hf",
        "loader": "auto",
        "images": "pil",
        "dtype": "bfloat16",
        "parser": "recovery",
        "role": "comparison",
    },
    "llava": {
        "hf_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "loader": "llava_onevision",
        "images": "pil",
        "dtype": "float16",
        "parser": "recovery",
        "role": "comparison",
    },
    "idefics2": {
        "hf_id": "HuggingFaceM4/idefics2-8b",
        "loader": "auto",
        "images": "pil",
        "dtype": "float16",
        "parser": "recovery",
        "role": "comparison",
    },
}

# Independent atomic prediction was only run for the two pipeline models.
METHODS_BY_VLM = {
    "qwen3": ["baseline", "joint", "selfdecompose", "independent"],
    "internvl": ["baseline", "joint", "selfdecompose", "independent"],
}
DEFAULT_METHODS = ["baseline"]

OUTPUT_DIRS = {
    "qwen3": "qwen3_predictions",
    "internvl": "internvl_predictions",
    "qwen3vl_32b": "qwen3vl_32b_predictions",
    "qwen2vl_2b": "qwen2vl_2b_predictions",
    "internvl3_1b": "internvl3_1b_predictions",
    "llava": "llava_onevision_predictions",
    "idefics2": "idefics2_predictions",
}

# Short names used by the hypothesis-bias output folders.
BIAS_DIRS = {"qwen3": "qwen3", "internvl": "internvl"}

# The bias check covers full-hypothesis and joint atomic prediction only.
BIAS_METHODS = ["baseline", "joint"]


def methods_for(vlm_key):
    return METHODS_BY_VLM.get(vlm_key, DEFAULT_METHODS)


def image_path(img_id):
    return os.path.join(config.IMAGE_DIR, f"{img_id}.jpg")


def _loader_class(name):
    if name == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    if name == "qwen2vl":
        from transformers import Qwen2VLForConditionalGeneration
        return Qwen2VLForConditionalGeneration
    if name == "llava_onevision":
        from transformers import LlavaOnevisionForConditionalGeneration
        return LlavaOnevisionForConditionalGeneration
    return AutoModelForImageTextToText


class VLMAdapter:
    def __init__(self, model, processor, images, parser="plain",
                 hf_id="", vlm_key=""):
        self.model = model
        self.processor = processor
        self.images = images
        self.parser = parser
        self.hf_id = hf_id
        self.vlm_key = vlm_key

    def load_image(self, img_id, blank=None):
        """blank is None for the real image, or 'black' / 'white' for the
        hypothesis-bias check."""
        if blank == "black":
            path = config.BLACK_IMAGE
        elif blank == "white":
            path = config.WHITE_IMAGE
        else:
            path = image_path(img_id)

        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        if self.images == "vision_info":
            return path
        return Image.open(path).convert("RGB")

    def render_prompt(self, image_ref, prompt_text):
        if self.images == "vision_info":
            image_entry = {"type": "image", "image": f"file://{image_ref}"}
        else:
            image_entry = {"type": "image"}

        messages = [{
            "role": "user",
            "content": [image_entry, {"type": "text", "text": prompt_text}],
        }]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _images_for_processor(self, image_ref, prompt_text):
        if self.images == "vision_info":
            from qwen_vl_utils import process_vision_info
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_ref}"},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            image_inputs, _ = process_vision_info(messages)
            return image_inputs
        return image_ref

    def make_inputs(self, image_ref, rendered_prompt, prompt_text=""):
        images = self._images_for_processor(image_ref, prompt_text)
        text = [rendered_prompt] if self.images == "vision_info" else rendered_prompt
        inputs = self.processor(
            text=text, images=images, return_tensors="pt"
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)
        return inputs

    def generate(self, image_ref, prompt_text):
        rendered = self.render_prompt(image_ref, prompt_text)
        inputs = self.make_inputs(image_ref, rendered, prompt_text)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS_VLM,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        return self.processor.batch_decode(
            generated_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def score_labels(self, image_ref, prompt_text):
        """Forced-choice scoring: how likely is each label as the continuation
        of the JSON prefix, given the same prompt and image."""
        rendered = self.render_prompt(image_ref, prompt_text)
        label_prefix = '{"label": "'

        prefix_inputs = self.make_inputs(
            image_ref, rendered + label_prefix, prompt_text
        )
        prefix_len = prefix_inputs.input_ids.shape[1]

        raw_scores = {}
        for label in config.FINAL_LABELS:
            inputs = self.make_inputs(
                image_ref, rendered + label_prefix + label, prompt_text
            )
            full_len = inputs.input_ids.shape[1]
            label_len = max(full_len - prefix_len, 1)

            labels = inputs.input_ids.clone()
            labels[:, :prefix_len] = -100

            with torch.no_grad():
                out = self.model(**inputs, labels=labels)

            raw_scores[label] = -float(out.loss.item()) * label_len

        return raw_scores


def load(vlm_key):
    spec = VLM_MODELS[vlm_key]
    hf_id = spec["hf_id"]

    kwargs = {"trust_remote_code": True}
    if config.HF_TOKEN:
        kwargs["token"] = config.HF_TOKEN
    if config.HF_CACHE_DIR:
        kwargs["cache_dir"] = config.HF_CACHE_DIR

    dtype = getattr(torch, spec.get("dtype", "bfloat16"))

    print(f"Loading {hf_id} ...", flush=True)
    processor = AutoProcessor.from_pretrained(hf_id, **kwargs)
    model = _loader_class(spec["loader"]).from_pretrained(
        hf_id, dtype=dtype, device_map="auto", **kwargs
    ).eval()
    print("Model loaded.", flush=True)

    return VLMAdapter(model, processor, spec["images"],
                      parser=spec.get("parser", "plain"),
                      hf_id=hf_id, vlm_key=vlm_key)