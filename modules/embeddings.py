# modules/embeddings.py
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

# ----------------- CLIP for Images -----------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image):
    """
    image: PIL Image
    returns: numpy vector
    """
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.squeeze().numpy()

# ----------------- Dummy Text Embedding -----------------
# Replace with OpenAI / HuggingFace embeddings for production
def get_text_embedding(text):
    """
    Simple hash-based dummy embedding
    """
    vec = np.zeros(512, dtype=np.float32)
    for i, c in enumerate(text):
        vec[i % 512] += ord(c)
    vec /= (np.linalg.norm(vec) + 1e-6)
    return vec
