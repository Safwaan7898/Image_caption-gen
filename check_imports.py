from transformers import BlipForConditionalGeneration, BlipProcessor, DetrForObjectDetection, DetrImageProcessor
import torch
import streamlit
from PIL import Image
import utils, captioner
print("All imports OK")
print("torch:", torch.__version__)
print("streamlit:", streamlit.__version__)
