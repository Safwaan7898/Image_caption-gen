"""
app.py
------
Streamlit web UI for the AI Image Caption Generator.
Run with:  streamlit run app.py
"""

import io
import logging
import urllib.request
from pathlib import Path
from typing import Optional

import streamlit as st
from PIL import Image

from captioner import (
    generate_caption,
    detect_objects,
    load_blip_model,
    load_detr_model,
)
from utils import (
    draw_bounding_boxes,
    open_image_from_bytes,
    preprocess_image,
    validate_image_file,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
)

# ── Sample image URL (public-domain photo from Wikimedia) ────────────────────
SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/"
    "Cat03.jpg/320px-Cat03.jpg"
)
SAMPLE_IMAGE_PATH = Path("sample_images/sample_cat.jpg")


# ── Cached model loaders (run only once per session) ─────────────────────────

@st.cache_resource(show_spinner="Loading BLIP captioning model…")
def get_blip_model():
    """Load and cache the BLIP model."""
    return load_blip_model()


@st.cache_resource(show_spinner="Loading DETR object-detection model…")
def get_detr_model():
    """Load and cache the DETR model."""
    return load_detr_model()


# ── Helper: download sample image ────────────────────────────────────────────

def get_sample_image() -> Optional[Image.Image]:
    """
    Return a sample PIL Image.
    First tries to load from disk, then downloads from the web,
    then falls back to a generated placeholder if offline.
    """
    # 1. Try local file
    if SAMPLE_IMAGE_PATH.exists():
        try:
            return Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
        except Exception:
            pass

    # 2. Try downloading
    try:
        SAMPLE_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_IMAGE_URL, SAMPLE_IMAGE_PATH)
        return Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    except Exception:
        pass

    # 3. Offline fallback: generate a simple gradient placeholder
    try:
        from PIL import ImageDraw
        img = Image.new("RGB", (320, 240), color=(100, 149, 237))
        draw = ImageDraw.Draw(img)
        draw.text((60, 100), "Sample Image\n(offline fallback)", fill="white")
        return img
    except Exception:
        return None


# ── Session history helpers ───────────────────────────────────────────────────

def add_to_history(image: Image.Image, caption: str) -> None:
    """Add a result to session history, keeping only the last 5 entries."""
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Create a small thumbnail for display
    thumb = image.copy()
    thumb.thumbnail((120, 120))
    st.session_state["history"].insert(0, {"thumbnail": thumb, "caption": caption})
    st.session_state["history"] = st.session_state["history"][:5]


def render_history() -> None:
    """Render the last 5 session results at the bottom of the page."""
    history = st.session_state.get("history", [])
    if not history:
        return

    st.markdown("---")
    st.subheader("🕘 Session History (last 5)")
    cols = st.columns(len(history))
    for col, entry in zip(cols, history):
        with col:
            st.image(entry["thumbnail"], use_container_width=True)
            st.caption(entry["caption"])


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Title & description ───────────────────────────────────────────────────
    st.title("🖼️ AI Image Caption Generator")
    st.markdown(
        "Upload an image and let a **vision-language AI model** describe what it sees. "
        "Powered by [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) "
        "and [DETR](https://huggingface.co/facebook/detr-resnet-50) — 100 % local, no API keys."
    )
    st.markdown("---")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Caption Controls")

        num_beams = st.slider(
            "Beam search width",
            min_value=1, max_value=8, value=4,
            help="Higher = better quality captions but slower inference.",
        )
        max_length = st.slider(
            "Max caption length (tokens)",
            min_value=20, max_value=150, value=50,
        )
        generate_alternatives = st.checkbox(
            "Generate 3 alternative captions", value=False
        )
        text_prefix = st.text_input(
            "Conditional prefix (optional)",
            placeholder='e.g. "a photograph of"',
            help="Guide the model by providing a starting phrase.",
        )

        st.markdown("---")
        st.header("🔍 Object Detection")
        run_detection = st.checkbox("Enable object detection (DETR)", value=True)
        draw_boxes = st.checkbox("Draw bounding boxes", value=True)
        score_threshold = st.slider(
            "Confidence threshold",
            min_value=0.5, max_value=1.0, value=0.8, step=0.05,
        )

    # ── Image source selection ────────────────────────────────────────────────
    source = st.radio(
        "Image source",
        ["Upload an image", "Try a sample image"],
        horizontal=True,
    )

    raw_image: Optional[Image.Image] = None

    if source == "Upload an image":
        uploaded = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            file_bytes = uploaded.read()
            valid, err_msg = validate_image_file(file_bytes, uploaded.name)
            if not valid:
                st.error(f"❌ {err_msg}")
                return
            try:
                raw_image = open_image_from_bytes(file_bytes)
            except ValueError as exc:
                st.error(f"❌ Could not open image: {exc}")
                return
    else:
        with st.spinner("Loading sample image…"):
            raw_image = get_sample_image()
        if raw_image is None:
            st.error("❌ Could not load sample image. Please upload your own.")
            return
        st.info("ℹ️ Using a sample cat photo (public domain, Wikimedia Commons).")

    # ── Display uploaded / sample image ──────────────────────────────────────
    if raw_image is None:
        st.info("👆 Upload an image or choose the sample to get started.")
        return

    # ── Generate button ───────────────────────────────────────────────────────
    if not st.button("🚀 Generate Caption", type="primary"):
        # Still show the image even before the button is pressed
        st.image(raw_image, caption="Your image", use_container_width=False, width=400)
        return

    # ── Preprocessing ─────────────────────────────────────────────────────────
    processed_image = preprocess_image(raw_image)

    # ── Load models ───────────────────────────────────────────────────────────
    try:
        blip_processor, blip_model = get_blip_model()
    except RuntimeError as exc:
        st.error(
            f"❌ Model loading failed.\n\n{exc}\n\n"
            "**Fix:** Make sure you are connected to the internet on first run so "
            "Hugging Face can download the model weights (~1 GB)."
        )
        return

    # ── Caption generation ────────────────────────────────────────────────────
    num_captions = 3 if generate_alternatives else 1

    with st.spinner("✍️ Generating caption…"):
        try:
            captions, caption_time = generate_caption(
                processed_image,
                blip_processor,
                blip_model,
                num_beams=num_beams,
                max_length=max_length,
                text_prefix=text_prefix or None,
                num_captions=num_captions,
            )
        except Exception as exc:
            st.error(f"❌ Caption generation failed: {exc}")
            logger.exception("Caption generation error")
            return

    main_caption = captions[0]

    # ── Object detection ──────────────────────────────────────────────────────
    detections = []
    detection_time = 0.0

    if run_detection:
        try:
            detr_processor, detr_model = get_detr_model()
        except RuntimeError as exc:
            st.warning(f"⚠️ Object detection model failed to load: {exc}")
            run_detection = False

    if run_detection:
        with st.spinner("🔍 Detecting objects…"):
            try:
                detections, detection_time = detect_objects(
                    processed_image,
                    detr_processor,  # type: ignore[possibly-undefined]
                    detr_model,      # type: ignore[possibly-undefined]
                    score_threshold=score_threshold,
                )
            except Exception as exc:
                st.warning(f"⚠️ Object detection failed: {exc}")
                logger.exception("Object detection error")

    # ── Layout: image left, results right ────────────────────────────────────
    col_img, col_results = st.columns([1, 1], gap="large")

    with col_img:
        st.subheader("📷 Image")
        if run_detection and draw_boxes and detections:
            annotated = draw_bounding_boxes(processed_image, detections, score_threshold)
            st.image(annotated, caption="With bounding boxes", use_container_width=True)
        else:
            st.image(processed_image, use_container_width=True)

    with col_results:
        st.subheader("📝 Caption")
        st.success(f"**{main_caption}**")

        # Alternative captions
        if generate_alternatives and len(captions) > 1:
            with st.expander("🔄 Alternative captions"):
                for i, cap in enumerate(captions[1:], start=2):
                    st.write(f"{i}. {cap}")

        # Download caption as .txt
        caption_bytes = "\n".join(captions).encode("utf-8")
        st.download_button(
            label="⬇️ Download caption as .txt",
            data=caption_bytes,
            file_name="caption.txt",
            mime="text/plain",
        )

        # Copy-friendly display
        st.text_area("📋 Copy caption", value=main_caption, height=80)

        # Detected objects
        if run_detection and detections:
            st.subheader("🏷️ Detected Objects")
            # Show as a small table
            table_data = [
                {"Object": d["label"], "Confidence": f"{d['score']:.0%}"}
                for d in detections
            ]
            st.table(table_data)
        elif run_detection:
            st.info("No objects detected above the confidence threshold.")

        # Processing time
        total_time = caption_time + detection_time
        st.markdown(
            f"⏱️ Caption: `{caption_time:.2f}s` | "
            f"Detection: `{detection_time:.2f}s` | "
            f"Total: `{total_time:.2f}s`"
        )

    # ── Add to session history ────────────────────────────────────────────────
    add_to_history(processed_image, main_caption)

    # ── Session history ───────────────────────────────────────────────────────
    render_history()

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:grey; font-size:0.85em;'>"
        "AI Image Caption Generator · Built with ❤️ using BLIP & DETR · "
        "<b>#incodevision</b>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
