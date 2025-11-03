# modules/embeddings.py
import numpy as np
import torch
import streamlit as st
from transformers import CLIPProcessor, CLIPModel
# NEW: Import for semantic embeddings
from sentence_transformers import SentenceTransformer 

# ----------------- Text Embedding Setup -----------------
# CRITICAL FIX: Load the semantic model once.
try:
    # Use a small, efficient, high-performance model suitable for RAG
    # This model provides 384-dimensional vectors.
    text_model = SentenceTransformer('all-MiniLM-L6-v2') 
    TEXT_EMBED_DIM = text_model.get_sentence_embedding_dimension() # 384
    # Set the flag to true if the model loads successfully
    TEXT_MODEL_LOADED = True
    
except Exception as e:
    # If the model fails to load (e.g., due to file download issues), 
    # we default the dimension to 512 and use the DUMMY function.
    st.error(f"FATAL: Failed to load semantic model. RAG will use DUMMY EMBEDDINGS ({e})")
    TEXT_EMBED_DIM = 512
    text_model = None
    TEXT_MODEL_LOADED = False


def get_text_embedding(text):
    """
    Returns a semantic embedding (384-dim) or the dummy fallback (512-dim).
    """
    if TEXT_MODEL_LOADED:
        # Compute semantic embedding
        embedding = text_model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)
    else:
        # Fallback to original dummy embedding if model failed to load
        vec = np.zeros(TEXT_EMBED_DIM, dtype=np.float32)
        for i, c in enumerate(text):
            vec[i % TEXT_EMBED_DIM] += ord(c)
        vec /= (np.linalg.norm(vec) + 1e-6)
        return vec

# ----------------- CLIP for Images -----------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image_path):
    """
    image_path: Path object (or string path)
    returns: numpy vector (512-dim for CLIP base)
    """
    from PIL import Image 
    img = Image.open(image_path).convert("RGB")
    
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding.squeeze().numpy()