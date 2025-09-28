# modules/vector_db.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorDB:
    def __init__(self, dim=512):
        self.dim = dim
        self.vectors = []
        self.texts = []

    def add_vector(self, vec, text):
        self.vectors.append(vec)
        self.texts.append(text)

    def query_top_k(self, query_text, k=3, embedding_func=None):
        """
        Query using text input (default embedding function is get_text_embedding)
        """
        if embedding_func is None:
            from modules.embeddings import get_text_embedding
            embedding_func = get_text_embedding

        query_vec = embedding_func(query_text).reshape(1, -1)
        return self.query_top_k_embedding(query_vec, k)

    def query_top_k_embedding(self, query_vec, k=3):
        """
        Query using a precomputed embedding vector
        """
        vectors_np = np.array(self.vectors)
        if len(vectors_np) == 0:
            return []

        sims = cosine_similarity(query_vec.reshape(1, -1), vectors_np)[0]
        top_indices = sims.argsort()[::-1][:k]
        return [self.texts[i] for i in top_indices]
