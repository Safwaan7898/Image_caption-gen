"""
cli.py
------
Command-line interface for the AI Image Caption Generator.

Usage:
    python cli.py path/to/image.jpg
    python cli.py path/to/image.jpg --beams 6 --max-length 80 --prefix "a photo of"
    python cli.py path/to/image.jpg --no-detection
"""

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image

from captioner import (
    generate_caption,
    detect_objects,
    load_blip_model,
    load_detr_model,
)
from utils import preprocess_image, validate_image_file

# Show INFO-level logs in the terminal so the user can see model loading progress
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a caption for an image using BLIP + DETR."
    )
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument(
        "--beams", type=int, default=4, help="Beam search width (default: 4)."
    )
    parser.add_argument(
        "--max-length", type=int, default=50, help="Max caption tokens (default: 50)."
    )
    parser.add_argument(
        "--prefix", type=str, default="", help='Optional text prefix, e.g. "a photo of".'
    )
    parser.add_argument(
        "--alternatives", type=int, default=1,
        help="Number of captions to generate (default: 1).",
    )
    parser.add_argument(
        "--no-detection", action="store_true", help="Skip object detection."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Object detection confidence threshold (default: 0.8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path)

    # ── Validate file ─────────────────────────────────────────────────────────
    if not image_path.exists():
        print(f"❌ File not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    file_bytes = image_path.read_bytes()
    valid, err_msg = validate_image_file(file_bytes, image_path.name)
    if not valid:
        print(f"❌ {err_msg}", file=sys.stderr)
        sys.exit(1)

    # ── Load and preprocess image ─────────────────────────────────────────────
    try:
        raw_image = Image.open(image_path)
        processed = preprocess_image(raw_image)
    except Exception as exc:
        print(f"❌ Could not open image: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Load BLIP and generate caption ────────────────────────────────────────
    print("\n🔄 Loading BLIP captioning model…")
    try:
        blip_processor, blip_model = load_blip_model()
    except RuntimeError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        sys.exit(1)

    print("✍️  Generating caption…")
    captions, caption_time = generate_caption(
        processed,
        blip_processor,
        blip_model,
        num_beams=args.beams,
        max_length=args.max_length,
        text_prefix=args.prefix or None,
        num_captions=args.alternatives,
    )

    print(f"\n{'─'*50}")
    print("📝 CAPTION(S)")
    print(f"{'─'*50}")
    for i, cap in enumerate(captions, start=1):
        print(f"  {i}. {cap}")
    print(f"\n  ⏱️  Generated in {caption_time:.2f}s")

    # ── Object detection ──────────────────────────────────────────────────────
    if not args.no_detection:
        print("\n🔄 Loading DETR object-detection model…")
        try:
            detr_processor, detr_model = load_detr_model()
        except RuntimeError as exc:
            print(f"⚠️  Object detection skipped: {exc}", file=sys.stderr)
            return

        print("🔍 Detecting objects…")
        detections, det_time = detect_objects(
            processed, detr_processor, detr_model, score_threshold=args.threshold
        )

        print(f"\n{'─'*50}")
        print("🏷️  DETECTED OBJECTS")
        print(f"{'─'*50}")
        if detections:
            for det in detections:
                print(f"  • {det['label']:<20} {det['score']:.0%}")
        else:
            print("  No objects detected above the confidence threshold.")
        print(f"\n  ⏱️  Detected in {det_time:.2f}s")

    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
