"""
Data loading utilities for the EVG Course Recommender.
Downloads and parses the official EVG course catalog CSV.
"""

import os
import pandas as pd
import requests

from utils.preprocessing import clean_text, compile_text, STOP_WORDS

CATALOG_URL = "https://www.escolavirtual.gov.br/catalogo/exportar/csv"
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "catalogo_evg.csv")

# Columns used to build the text representation for embeddings
TEXT_COLS = ["nome_curso", "eixos_tematicos", "competencias", "apresentacao", "conteudo_programatico"]


def download_catalog(dest_path: str = DEFAULT_CSV_PATH) -> str:
    """Download the latest EVG catalog CSV and save to *dest_path*."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Baixando catálogo da EVG de {CATALOG_URL} …")
    resp = requests.get(CATALOG_URL, timeout=60)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)
    print(f"Catálogo salvo em {dest_path} ({len(resp.content)} bytes).")
    return dest_path


def load_catalog(path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Read the pipe-delimited EVG catalog CSV into a DataFrame.

    Returns a DataFrame with an extra ``compilado_textual`` column
    containing all text columns cleaned and merged.
    """
    if not os.path.exists(path):
        download_catalog(path)

    df = pd.read_csv(path, sep="|", encoding="utf-8", dtype=str).fillna("")

    # Clean each text column
    for col in TEXT_COLS:
        if col in df.columns:
            df[col + "_clean"] = df[col].apply(lambda t: clean_text(t, STOP_WORDS))

    # Build the compiled text column
    clean_cols = [c + "_clean" for c in TEXT_COLS if c in df.columns]
    df["compilado_textual"] = df.apply(lambda row: compile_text(row, clean_cols), axis=1)

    print(f"Catálogo carregado: {len(df)} cursos.")
    return df
