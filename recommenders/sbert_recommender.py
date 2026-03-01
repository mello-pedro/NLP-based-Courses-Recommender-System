"""
SBERT-based course recommender — cosine similarity only.
Adapted from the original BERT_recomend_construct.py.
"""

import os
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from recommenders.base import BaseRecommender
from utils.preprocessing import clean_text, STOP_WORDS

MODEL_NAME = "all-MiniLM-L6-v2"


class SBERTRecommender(BaseRecommender):
    """Sentence-BERT recommender using cosine similarity."""

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.model_name = model_name
        self.model: SentenceTransformer | None = None
        self.embeddings: torch.Tensor | None = None

    # ── Build ────────────────────────────────────────────────────────────
    def build_embeddings(self, batch_size: int = 64) -> None:
        """Encode ``compilado_textual`` column into normalised embeddings."""
        if self.df is None or "compilado_textual" not in self.df.columns:
            raise ValueError("Chame load_data() com um DataFrame contendo 'compilado_textual' antes.")

        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

        texts = self.df["compilado_textual"].tolist()
        all_embs: list[torch.Tensor] = []

        with torch.no_grad():
            for start in tqdm(range(0, len(texts), batch_size), desc="SBERT embeddings"):
                batch = texts[start : start + batch_size]
                embs = self.model.encode(batch, convert_to_tensor=True, normalize_embeddings=True)
                all_embs.append(embs)

        self.embeddings = torch.cat(all_embs, dim=0)
        print(f"SBERT: {self.embeddings.shape[0]} embeddings gerados ({self.embeddings.shape[1]}d).")

    # ── Persist ──────────────────────────────────────────────────────────
    def save_embeddings(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.embeddings, path)
        print(f"SBERT embeddings salvos em {path}")

    def load_embeddings(self, path: str) -> None:
        self.embeddings = torch.load(path, map_location="cpu", weights_only=True)
        print(f"SBERT embeddings carregados de {path} ({self.embeddings.shape[0]} cursos).")

    # ── Recommend ────────────────────────────────────────────────────────
    def recommend(
        self, query_text: str, top_n: int = 3, threshold: float = 0.0
    ) -> pd.DataFrame:
        """Cosine-similarity search between the user query and all courses."""
        if self.embeddings is None:
            raise ValueError("Embeddings não carregados. Chame build_embeddings() ou load_embeddings().")
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

        cleaned = clean_text(query_text, STOP_WORDS)
        query_emb = self.model.encode(cleaned, convert_to_tensor=True, normalize_embeddings=True).cpu()
        catalog_emb = self.embeddings.cpu()

        scores = util.cos_sim(query_emb, catalog_emb).squeeze().tolist()

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
