"""
utils.py
--------
Image utility functions: validation, EXIF rotation fix, resizing, and bounding-box drawing.
Kept separate from model code so they can be reused anywhere.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont, ExifTags

logger = logging.getLogger(__name__)

# Maximum allowed upload size in bytes (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Supported image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Resize images larger than this before sending to the model (keeps inference fast)
MAX_INFERENCE_SIZE = 1024


def validate_image_file(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Check that the uploaded file is within size limits and has an allowed extension.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to check the extension).

    Returns:
        (True, "") if valid, or (False, error_message) if not.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Please upload jpg, jpeg, png, or webp."

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        return False, f"File is {size_mb:.1f} MB. Maximum allowed size is 10 MB."

    return True, ""


def fix_exif_rotation(image: Image.Image) -> Image.Image:
    """
    Rotate the image to its correct orientation using EXIF metadata.
    Many phone photos are stored rotated; this corrects that before inference.

    Args:
        image: A PIL Image object.

    Returns:
        The correctly oriented PIL Image.
    """
    try:
        # Find the EXIF tag number for 'Orientation'
        orientation_tag = next(
            key for key, val in ExifTags.TAGS.items() if val == "Orientation"
        )
        exif = image._getexif()  # type: ignore[attr-defined]
        if exif is None:
            return image

        orientation = exif.get(orientation_tag)
        rotation_map = {3: 180, 6: 270, 8: 90}
        if orientation in rotation_map:
            image = image.rotate(rotation_map[orientation], expand=True)
    except (AttributeError, StopIteration, Exception):
        # Not all images have EXIF data — that's fine, just return as-is
        pass
    return image


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Full preprocessing pipeline:
      1. Fix EXIF rotation.
      2. Convert to RGB (removes alpha channel, handles palette images).
      3. Resize if larger than MAX_INFERENCE_SIZE on either dimension.

    Args:
        image: Raw PIL Image from upload.

    Returns:
        Preprocessed PIL Image ready for the model.
    """
    image = fix_exif_rotation(image)
    image = image.convert("RGB")

    # Resize while keeping aspect ratio
    w, h = image.size
    if max(w, h) > MAX_INFERENCE_SIZE:
        scale = MAX_INFERENCE_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        logger.debug("Resized image from (%d, %d) to (%d, %d)", w, h, new_w, new_h)

    return image


def draw_bounding_boxes(
    image: Image.Image,
    detections: List[dict],
    score_threshold: float = 0.8,
) -> Image.Image:
    """
    Draw bounding boxes and labels on a copy of the image.

    Args:
        image:           PIL Image to annotate.
        detections:      List of dicts with keys 'label', 'score', 'box'
                         where 'box' is {'xmin', 'ymin', 'xmax', 'ymax'}.
        score_threshold: Only draw boxes with confidence >= this value.

    Returns:
        A new PIL Image with boxes drawn on it.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # Try to load a small font; fall back to the default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        if det["score"] < score_threshold:
            continue
        box = det["box"]
        label = f"{det['label']} {det['score']:.0%}"
        coords = [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
        draw.rectangle(coords, outline="red", width=2)
        draw.text((box["xmin"] + 4, box["ymin"] + 2), label, fill="red", font=font)

    return annotated


def open_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """
    Open a PIL Image from raw bytes.

    Args:
        file_bytes: Raw bytes of the image file.

    Returns:
        PIL Image object.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    import io

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # Force decode so corrupt files raise here, not later
        return image
    except Exception as exc:
        raise ValueError(f"Could not open image: {exc}") from exc
