#!/usr/bin/env python3
"""
Build (or rebuild) embeddings for both SBERT and TF-IDF recommenders.
Run this script whenever the catalog CSV is updated.

Usage:
    python scripts/build_embeddings.py
"""

import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import download_catalog, load_catalog
from recommenders.sbert_recommender import SBERTRecommender
from recommenders.tfidf_recommender import TFIDFRecommender

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "catalogo_evg.csv")
SBERT_EMB_PATH = os.path.join(BASE_DIR, "embeddings", "sbert_embeddings.pt")
TFIDF_EMB_PATH = os.path.join(BASE_DIR, "embeddings", "tfidf_artifacts.pkl")


def main() -> None:
    # 1. Download latest catalog
    download_catalog(DATA_PATH)

    # 2. Load & clean
    df = load_catalog(DATA_PATH)

    # 3. SBERT
    print("\n══════ SBERT ══════")
    sbert = SBERTRecommender()
    sbert.load_data(df)
    sbert.build_embeddings()
    sbert.save_embeddings(SBERT_EMB_PATH)

    # 4. TF-IDF
    print("\n══════ TF-IDF ══════")
    tfidf = TFIDFRecommender()
    tfidf.load_data(df)
    tfidf.build_embeddings()
    tfidf.save_embeddings(TFIDF_EMB_PATH)

    print("\n✅ Embeddings gerados com sucesso!")


if __name__ == "__main__":
    main()
