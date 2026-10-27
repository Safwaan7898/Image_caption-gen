# 🖼️ AI Image Caption Generator

> **Internship Project — Task 03 | #incodevision**
> Explore how AI interprets images using state-of-the-art vision-language models — 100 % local, no API keys required.

---

## 📌 Project Description

This project is a fully local AI-powered web application that takes an image as input and automatically generates a meaningful natural-language caption describing its content. It also detects the main objects in the image with confidence scores and optional bounding boxes.

Everything runs on your machine using free pre-trained models from Hugging Face. No cloud API, no subscription, no API key.

---

## ✨ Features

| Feature | Details |
|---|---|
| Image upload | JPG, JPEG, PNG, WEBP — up to 10 MB |
| Sample image | Auto-downloads a public-domain photo; offline fallback included |
| Caption generation | BLIP vision-language model with beam search |
| Alternative captions | Generate 3 variations in one click |
| Conditional captioning | Guide the model with a text prefix |
| Object detection | DETR model lists objects + confidence scores |
| Bounding boxes | Drawn on a second copy of the image |
| Download caption | Save as `.txt` with one click |
| Session history | Last 5 results shown as thumbnails |
| CLI | `python cli.py image.jpg` for terminal use |
| Performance | Models cached, large images auto-resized, EXIF rotation fixed |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web UI
- **Hugging Face Transformers** — model loading and inference
- **BLIP** (`Salesforce/blip-image-captioning-base`) — image captioning
- **DETR** (`facebook/detr-resnet-50`) — object detection
- **PyTorch** — deep learning backend (CPU or GPU)
- **Pillow** — image loading, preprocessing, annotation

---

## 🧠 How It Works

```
Image
  │
  ▼
Preprocessing (EXIF fix → RGB conversion → resize)
  │
  ▼
Vision Encoder (BLIP's ViT — splits image into patches, extracts features)
  │
  ▼
Language Decoder (BLIP's transformer decoder — generates tokens one by one)
  │
  ▼
Beam Search (explores multiple caption candidates, picks the best)
  │
  ▼
Caption  ──────────────────────────────────────────────────────────────────┐
                                                                           │
Image ──► DETR CNN Backbone ──► Transformer Encoder/Decoder ──► Objects   │
                                                                           │
                                                                           ▼
                                                                    Final Result
```

---

## 📚 Key Concepts

**Computer Vision** — the field of AI that enables machines to interpret and understand visual information from images and videos.

**Vision-Language Models (VLMs)** — models that bridge computer vision and natural language processing, allowing them to describe images in text.

**BLIP (Bootstrapping Language-Image Pre-training)** — a VLM from Salesforce that uses a Vision Transformer (ViT) as an image encoder and a transformer decoder to generate captions. It was pre-trained on hundreds of millions of image-text pairs.

**Transformers** — the neural network architecture behind modern AI. They use "attention" mechanisms to weigh the importance of different parts of the input when making predictions.

**Beam Search** — a decoding strategy that keeps track of the top-N most likely sequences at each step, producing higher-quality captions than greedy (pick-the-best-at-each-step) decoding.

**Object Detection** — locating and classifying multiple objects within an image, returning bounding boxes and class labels.

**DETR (Detection Transformer)** — Facebook's end-to-end object detector that applies the transformer architecture directly to detection, eliminating the need for hand-crafted anchor boxes.

---

## 📁 Project Structure

```
task-3mage/
├── app.py               # Streamlit web UI
├── captioner.py         # Model loading, caption generation, object detection
├── utils.py             # Image validation, EXIF fix, resizing, bounding boxes
├── cli.py               # Command-line interface
├── test_captioner.py    # Pytest test suite
├── requirements.txt     # Pinned dependencies
├── README.md            # This file
├── .gitignore
└── sample_images/
    └── README.md        # Instructions for test images
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/task-3mage.git
cd task-3mage
```

### 2. Create a virtual environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **First run note:** The BLIP model (~990 MB) and DETR model (~160 MB) are downloaded automatically from Hugging Face the first time you run the app. Make sure you have an internet connection and ~1.5 GB of free disk space.

---

## 🚀 Running the App

### Web UI (Streamlit)

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Command-Line Interface

```bash
# Basic usage
python cli.py sample_images/animal.jpg

# With options
python cli.py sample_images/street.jpg --beams 6 --max-length 80 --prefix "a photograph of"

# Generate 3 captions, skip object detection
python cli.py sample_images/food.jpg --alternatives 3 --no-detection

# All options
python cli.py --help
```

---

## 🧪 Running Tests

```bash
# Run all tests (includes model-loading tests — requires internet + ~1 GB)
pytest test_captioner.py -v

# Skip model-heavy tests (fast, no download needed)
pytest test_captioner.py -v -m "not model"
```

---

## 📊 Sample Output

| Image | Generated Caption |
|---|---|
| 🐱 Cat on a sofa | "a cat sitting on a couch looking at the camera" |
| 🍕 Pizza on a table | "a pizza sitting on top of a wooden table" |
| 🏙️ City street | "a busy city street with cars and people walking" |
| 🐕 Dog in a park | "a dog running through a grassy field" |
| 🍎 Apple on a desk | "a red apple sitting on top of a wooden desk" |

---

## 📸 Screenshots

> Add screenshots of the running app here after your first run.
> Suggested: upload image → caption result → bounding boxes view → session history.

---

## ⚠️ Limitations

- **First run downloads ~1.15 GB** of model weights from Hugging Face. Subsequent runs use the local cache.
- **CPU inference is slow** — expect 5–30 seconds per image depending on your hardware. A GPU reduces this to under 1 second.
- **Captions can be wrong** — the model may hallucinate details or miss context, especially for unusual or abstract images.
- **English only** — the base BLIP model generates captions in English only.
- **Object detection misses small objects** — DETR works best on objects that occupy a reasonable portion of the image.

---

## 🔮 Future Improvements

- 🌍 **Multilingual captions** — use a translation model to output captions in any language
- 🔊 **Text-to-speech** — read the caption aloud using `gTTS` or `pyttsx3`
- 🎬 **Video captioning** — extract frames and caption each one
- 🔬 **Larger models** — swap in `Salesforce/blip2-opt-2.7b` or `Salesforce/blip-image-captioning-large` for better accuracy
- 📱 **Mobile-friendly UI** — responsive layout for phone uploads
- 🗂️ **Batch processing** — caption an entire folder of images at once

---

## 📄 License

This project is for educational purposes as part of an internship program.

---

*Built with ❤️ using BLIP, DETR, Streamlit, and Hugging Face Transformers.*
*#incodevision*
