"""
Abstract base class for course recommenders.
"""

from abc import ABC, abstractmethod
import pandas as pd


class BaseRecommender(ABC):
    """Common interface shared by SBERT and TF-IDF recommenders."""

    def __init__(self):
        self.df: pd.DataFrame | None = None

    def load_data(self, df: pd.DataFrame) -> None:
        """Receive the pre-cleaned catalog DataFrame."""
        self.df = df.copy()

    @abstractmethod
    def build_embeddings(self) -> None:
        """Generate vector representations from ``compilado_textual``."""

    @abstractmethod
    def save_embeddings(self, path: str) -> None:
        """Persist embeddings / model artifacts to disk."""

    @abstractmethod
    def load_embeddings(self, path: str) -> None:
        """Load pre-built embeddings / model artifacts from disk."""

    @abstractmethod
    def recommend(
        self, query_text: str, top_n: int = 3, threshold: float = 0.0
    ) -> pd.DataFrame:
        """Return a DataFrame of recommendations for a free-text query.

        Columns: ``nome_curso``, ``similaridade``, plus any other metadata.
        """
