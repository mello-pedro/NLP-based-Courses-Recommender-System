"""
TF-IDF-based course recommender — cosine similarity only.
Adapted from the original TFIDF_recomend_construct.py.
"""

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from recommenders.base import BaseRecommender
from utils.preprocessing import clean_text, STOP_WORDS


class TFIDFRecommender(BaseRecommender):
    """TF-IDF recommender using cosine similarity."""

    def __init__(self):
        super().__init__()
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    # ── Build ────────────────────────────────────────────────────────────
    def build_embeddings(self) -> None:
        """Fit TF-IDF vectorizer on ``compilado_textual`` and store the matrix."""
        if self.df is None or "compilado_textual" not in self.df.columns:
            raise ValueError("Chame load_data() com um DataFrame contendo 'compilado_textual' antes.")

        with tqdm(total=1, desc="TF-IDF embeddings") as pbar:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.df["compilado_textual"])
            pbar.update(1)

        print(f"TF-IDF: matriz {self.tfidf_matrix.shape[0]}×{self.tfidf_matrix.shape[1]} gerada.")

    # ── Persist ──────────────────────────────────────────────────────────
    def save_embeddings(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.tfidf_matrix}, f)
        print(f"TF-IDF artefatos salvos em {path}")

    def load_embeddings(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["matrix"]
        print(f"TF-IDF artefatos carregados de {path} ({self.tfidf_matrix.shape[0]} cursos).")

    # ── Recommend ────────────────────────────────────────────────────────
    def recommend(
        self, query_text: str, top_n: int = 3, threshold: float = 0.0
    ) -> pd.DataFrame:
        """Cosine-similarity search between the user query and all courses."""
        if self.tfidf_matrix is None:
            raise ValueError("Embeddings não carregados. Chame build_embeddings() ou load_embeddings().")

        cleaned = clean_text(query_text, STOP_WORDS)
        query_vec = self.vectorizer.transform([cleaned])

        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten().tolist()

        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        filtered = [(i, s) for i, s in scored if s >= threshold][:top_n]

        if not filtered:
            return pd.DataFrame(columns=["nome_curso", "similaridade", "apresentacao", "carga_horaria"])

        idxs = [i for i, _ in filtered]
        sims = [s for _, s in filtered]

        result = pd.DataFrame({
            "nome_curso": self.df.iloc[idxs]["nome_curso"].values,
            "similaridade": [round(s, 4) for s in sims],
            "apresentacao": self.df.iloc[idxs]["apresentacao"].values,
            "carga_horaria": self.df.iloc[idxs]["carga_horaria"].values,
        })
        result.index = range(1, len(result) + 1)
        return result
