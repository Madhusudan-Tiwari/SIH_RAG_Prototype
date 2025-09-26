# modules/ingestion.py
from pathlib import Path
from docx import Document
import fitz  # PyMuPDF
from PIL import Image
import whisper

# ----------------- Audio Model -----------------
whisper_model = whisper.load_model("base")

# ----------------- Text Files -----------------
def process_text_file(file_path):
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text.strip()
    elif ext == ".docx":
        doc = Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text.strip()
    else:
        return ""

# ----------------- Image Files -----------------
def process_image_file(file_path):
    # Return PIL Image (embedding handled elsewhere)
    img = Image.open(file_path).convert("RGB")
    return img

# ----------------- Audio Files -----------------
def process_audio_file(file_path):
    result = whisper_model.transcribe(str(file_path))
    return result["text"].strip()
