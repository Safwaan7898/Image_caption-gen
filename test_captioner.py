"""
test_captioner.py
-----------------
Pytest tests for utils.py and captioner.py.

Run all tests:
    pytest test_captioner.py -v

Skip model-heavy tests (no internet / slow machine):
    pytest test_captioner.py -v -m "not model"
"""

import io
import pytest
from pathlib import Path
from PIL import Image

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_rgb_image() -> Image.Image:
    """A tiny 64x64 solid-colour RGB image for fast testing."""
    return Image.new("RGB", (64, 64), color=(120, 180, 60))


@pytest.fixture
def large_rgb_image() -> Image.Image:
    """A 2000x1500 image that should be resized by preprocess_image."""
    return Image.new("RGB", (2000, 1500), color=(200, 100, 50))


@pytest.fixture
def rgba_image() -> Image.Image:
    """An RGBA image to test RGB conversion."""
    return Image.new("RGBA", (64, 64), color=(10, 20, 30, 128))


# ── utils.py tests ────────────────────────────────────────────────────────────

class TestValidateImageFile:
    def test_valid_jpg(self):
        from utils import validate_image_file
        img = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        valid, msg = validate_image_file(buf.getvalue(), "photo.jpg")
        assert valid is True
        assert msg == ""

    def test_invalid_extension(self):
        from utils import validate_image_file
        valid, msg = validate_image_file(b"data", "file.bmp")
        assert valid is False
        assert "bmp" in msg.lower()

    def test_oversized_file(self):
        from utils import validate_image_file, MAX_FILE_SIZE_BYTES
        big_bytes = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        valid, msg = validate_image_file(big_bytes, "big.jpg")
        assert valid is False
        assert "10 MB" in msg or "MB" in msg

    def test_valid_png(self):
        from utils import validate_image_file
        img = Image.new("RGB", (10, 10))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        valid, msg = validate_image_file(buf.getvalue(), "image.png")
        assert valid is True


class TestPreprocessImage:
    def test_converts_rgba_to_rgb(self, rgba_image):
        from utils import preprocess_image
        result = preprocess_image(rgba_image)
        assert result.mode == "RGB"

    def test_resizes_large_image(self, large_rgb_image):
        from utils import preprocess_image, MAX_INFERENCE_SIZE
        result = preprocess_image(large_rgb_image)
        assert max(result.size) <= MAX_INFERENCE_SIZE

    def test_small_image_unchanged_size(self, small_rgb_image):
        from utils import preprocess_image
        result = preprocess_image(small_rgb_image)
        assert result.size == small_rgb_image.size

    def test_output_is_pil_image(self, small_rgb_image):
        from utils import preprocess_image
        result = preprocess_image(small_rgb_image)
        assert isinstance(result, Image.Image)


class TestOpenImageFromBytes:
    def test_valid_jpeg_bytes(self):
        from utils import open_image_from_bytes
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        result = open_image_from_bytes(buf.getvalue())
        assert isinstance(result, Image.Image)

    def test_invalid_bytes_raises(self):
        from utils import open_image_from_bytes
        with pytest.raises(ValueError):
            open_image_from_bytes(b"this is not an image")


class TestDrawBoundingBoxes:
    def test_returns_pil_image(self, small_rgb_image):
        from utils import draw_bounding_boxes
        detections = [
            {"label": "cat", "score": 0.95, "box": {"xmin": 5, "ymin": 5, "xmax": 50, "ymax": 50}}
        ]
        result = draw_bounding_boxes(small_rgb_image, detections, score_threshold=0.8)
        assert isinstance(result, Image.Image)

    def test_does_not_modify_original(self, small_rgb_image):
        from utils import draw_bounding_boxes
        import numpy as np
        original_arr = np.array(small_rgb_image).copy()
        detections = [
            {"label": "dog", "score": 0.9, "box": {"xmin": 0, "ymin": 0, "xmax": 30, "ymax": 30}}
        ]
        draw_bounding_boxes(small_rgb_image, detections)
        assert np.array_equal(np.array(small_rgb_image), original_arr)

    def test_below_threshold_not_drawn(self, small_rgb_image):
        from utils import draw_bounding_boxes
        detections = [
            {"label": "cat", "score": 0.5, "box": {"xmin": 5, "ymin": 5, "xmax": 50, "ymax": 50}}
        ]
        # Should not raise even if nothing is drawn
        result = draw_bounding_boxes(small_rgb_image, detections, score_threshold=0.9)
        assert isinstance(result, Image.Image)


# ── captioner.py tests (marked "model" — skip with -m "not model") ────────────

@pytest.mark.model
class TestGenerateCaption:
    """
    These tests actually load the BLIP model (~1 GB download on first run).
    Skip them on CI or slow machines with:  pytest -m "not model"
    """

    @pytest.fixture(scope="class")
    def blip(self):
        from captioner import load_blip_model
        return load_blip_model()

    def test_returns_list_of_strings(self, blip, small_rgb_image):
        from captioner import generate_caption
        processor, model = blip
        captions, elapsed = generate_caption(small_rgb_image, processor, model)
        assert isinstance(captions, list)
        assert len(captions) == 1
        assert isinstance(captions[0], str)
        assert len(captions[0]) > 0

    def test_elapsed_is_positive(self, blip, small_rgb_image):
        from captioner import generate_caption
        processor, model = blip
        _, elapsed = generate_caption(small_rgb_image, processor, model)
        assert elapsed > 0

    def test_alternative_captions_count(self, blip, small_rgb_image):
        from captioner import generate_caption
        processor, model = blip
        captions, _ = generate_caption(
            small_rgb_image, processor, model, num_captions=3
        )
        assert len(captions) == 3

    def test_prefix_does_not_crash(self, blip, small_rgb_image):
        from captioner import generate_caption
        processor, model = blip
        captions, _ = generate_caption(
            small_rgb_image, processor, model, text_prefix="a photo of"
        )
        assert isinstance(captions[0], str)


@pytest.mark.model
class TestDetectObjects:
    """
    These tests load the DETR model (~160 MB download on first run).
    Skip with:  pytest -m "not model"
    """

    @pytest.fixture(scope="class")
    def detr(self):
        from captioner import load_detr_model
        return load_detr_model()

    def test_returns_list(self, detr, small_rgb_image):
        from captioner import detect_objects
        processor, model = detr
        detections, elapsed = detect_objects(small_rgb_image, processor, model)
        assert isinstance(detections, list)
        assert elapsed > 0

    def test_detection_dict_keys(self, detr, small_rgb_image):
        from captioner import detect_objects
        processor, model = detr
        detections, _ = detect_objects(
            small_rgb_image, processor, model, score_threshold=0.0
        )
        if detections:
            d = detections[0]
            assert "label" in d
            assert "score" in d
            assert "box" in d
            assert all(k in d["box"] for k in ("xmin", "ymin", "xmax", "ymax"))
