# modules/vector_db.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorDB:
    def __init__(self, dim=512):
        self.dim = dim
        self.vectors = []  # list of np.array
        self.texts = []

    def add_vector(self, vec, text):
        # Ensure stored vector is np.array of correct shape
        vec_np = np.array(vec).reshape(1, -1)
        self.vectors.append(vec_np)
        self.texts.append(text)

    def query_top_k_embedding(self, query_vec, k=3):
        """
        Retrieve top-k texts given a query vector.
        query_vec: np.array of shape (dim,) or (1, dim)
        """
        if not self.vectors:
            return []

        query_vec = np.array(query_vec).reshape(1, -1)
        vectors_np = np.vstack(self.vectors)  # shape (n_vectors, dim)
        sims = cosine_similarity(query_vec, vectors_np)[0]
        top_indices = sims.argsort()[::-1][:k]
        return [self.texts[i] for i in top_indices]

    def query_top_k(self, query_text, k=3, embedding_func=None):
        """
        Retrieve top-k texts given query text. Optional embedding_func to convert text->vector.
        """
        if embedding_func is None:
            from modules.embeddings import get_text_embedding
            embedding_func = get_text_embedding

        query_vec = embedding_func(query_text)
        return self.query_top_k_embedding(query_vec, k=k)
