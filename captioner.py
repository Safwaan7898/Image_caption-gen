"""
captioner.py
------------
Model loading and inference logic.
- BLIP  : image captioning  (Salesforce/blip-image-captioning-base)
- DETR  : object detection  (facebook/detr-resnet-50)

Models are loaded once and cached; this module can be imported by both
the Streamlit app and the CLI without any UI dependency.
"""

import logging
import time
from typing import List, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    DetrForObjectDetection,
    DetrImageProcessor,
)

logger = logging.getLogger(__name__)

# ── Model identifiers ────────────────────────────────────────────────────────
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
DETR_MODEL_ID = "facebook/detr-resnet-50"

# Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", DEVICE)


# ── Model loaders (called once; Streamlit caches the return value) ───────────

def load_blip_model() -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """
    Download (first run) and load the BLIP captioning model.

    Returns:
        (processor, model) tuple ready for inference.

    Raises:
        RuntimeError: If the model cannot be downloaded or loaded.
    """
    try:
        logger.info("Loading BLIP model: %s", BLIP_MODEL_ID)
        processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
        model.to(DEVICE)
        model.eval()
        logger.info("BLIP model loaded successfully on %s", DEVICE)
        return processor, model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load BLIP model '{BLIP_MODEL_ID}'.\n"
            "Make sure you have an internet connection on first run so the model "
            "weights (~1 GB) can be downloaded from Hugging Face.\n"
            f"Original error: {exc}"
        ) from exc


def load_detr_model() -> Tuple[DetrImageProcessor, DetrForObjectDetection]:
    """
    Download (first run) and load the DETR object-detection model.

    Returns:
        (processor, model) tuple ready for inference.

    Raises:
        RuntimeError: If the model cannot be downloaded or loaded.
    """
    try:
        logger.info("Loading DETR model: %s", DETR_MODEL_ID)
        processor = DetrImageProcessor.from_pretrained(DETR_MODEL_ID)
        model = DetrForObjectDetection.from_pretrained(DETR_MODEL_ID)
        model.to(DEVICE)
        model.eval()
        logger.info("DETR model loaded successfully on %s", DEVICE)
        return processor, model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DETR model '{DETR_MODEL_ID}'.\n"
            "Make sure you have an internet connection on first run.\n"
            f"Original error: {exc}"
        ) from exc


# ── Caption generation ───────────────────────────────────────────────────────

def generate_caption(
    image: Image.Image,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    num_beams: int = 4,
    max_length: int = 50,
    text_prefix: Optional[str] = None,
    num_captions: int = 1,
) -> Tuple[List[str], float]:
    """
    Generate one or more captions for an image using BLIP.

    Args:
        image:        Preprocessed PIL Image (RGB).
        processor:    BLIP processor (tokenizer + feature extractor).
        model:        BLIP model.
        num_beams:    Beam-search width. Higher = better quality, slower.
        max_length:   Maximum number of tokens in the generated caption.
        text_prefix:  Optional prompt prefix, e.g. "a photograph of".
        num_captions: How many alternative captions to generate (uses
                      num_return_sequences with beam search).

    Returns:
        (captions, elapsed_seconds) where captions is a list of strings.
    """
    start = time.time()

    # Prepare inputs — pass the text prefix if provided (conditional captioning)
    if text_prefix and text_prefix.strip():
        inputs = processor(image, text=text_prefix.strip(), return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")

    # Move tensors to the same device as the model
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # num_return_sequences must be <= num_beams
    effective_beams = max(num_beams, num_captions)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            num_beams=effective_beams,
            max_length=max_length,
            num_return_sequences=num_captions,
            early_stopping=True,
        )

    # Decode token IDs back to human-readable text
    captions = processor.batch_decode(output_ids, skip_special_tokens=True)
    elapsed = time.time() - start
    logger.info("Generated %d caption(s) in %.2fs", num_captions, elapsed)
    return captions, elapsed


# ── Object detection ─────────────────────────────────────────────────────────

def detect_objects(
    image: Image.Image,
    processor: DetrImageProcessor,
    model: DetrForObjectDetection,
    score_threshold: float = 0.8,
) -> Tuple[List[dict], float]:
    """
    Detect objects in an image using DETR.

    Args:
        image:            Preprocessed PIL Image (RGB).
        processor:        DETR image processor.
        model:            DETR model.
        score_threshold:  Minimum confidence to include a detection.

    Returns:
        (detections, elapsed_seconds)
        Each detection is a dict: {'label': str, 'score': float, 'box': dict}
        where 'box' has keys xmin, ymin, xmax, ymax (pixel coordinates).
    """
    start = time.time()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: convert raw logits to boxes in original image coordinates
    target_sizes = torch.tensor([image.size[::-1]])  # (height, width)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=score_threshold
    )[0]

    detections = []
    for score, label_id, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label_name = model.config.id2label[label_id.item()]
        box_dict = {
            "xmin": int(box[0].item()),
            "ymin": int(box[1].item()),
            "xmax": int(box[2].item()),
            "ymax": int(box[3].item()),
        }
        detections.append(
            {"label": label_name, "score": round(score.item(), 4), "box": box_dict}
        )

    elapsed = time.time() - start
    logger.info("Detected %d object(s) in %.2fs", len(detections), elapsed)
    return detections, elapsed
