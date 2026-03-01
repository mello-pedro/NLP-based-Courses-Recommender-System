#!/usr/bin/env python3
"""
Que curso você gostaria de fazer na EVG?
─────────────────────────────────────────
Dash application entrypoint.
Recommends EVG courses via SBERT and TF-IDF (cosine similarity).
"""

import os
import sys

import dash
from dash import dcc, html, Input, Output, State
import pandas as pd

# ── Project root on sys.path ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from utils.data_loader import load_catalog, DEFAULT_CSV_PATH
from recommenders.sbert_recommender import SBERTRecommender
from recommenders.tfidf_recommender import TFIDFRecommender

SBERT_EMB_PATH = os.path.join(ROOT_DIR, "embeddings", "sbert_embeddings.pt")
TFIDF_EMB_PATH = os.path.join(ROOT_DIR, "embeddings", "tfidf_artifacts.pkl")

# ── Initialise models ──────────────────────────────────────────────────
print("Carregando catálogo…")
df_catalog = load_catalog(DEFAULT_CSV_PATH)

sbert = SBERTRecommender()
sbert.load_data(df_catalog)
if os.path.exists(SBERT_EMB_PATH):
    sbert.load_embeddings(SBERT_EMB_PATH)
else:
    print("⚠ SBERT embeddings não encontrados — gerando agora (pode demorar)…")
    sbert.build_embeddings()
    sbert.save_embeddings(SBERT_EMB_PATH)

tfidf = TFIDFRecommender()
tfidf.load_data(df_catalog)
if os.path.exists(TFIDF_EMB_PATH):
    tfidf.load_embeddings(TFIDF_EMB_PATH)
else:
    print("⚠ TF-IDF artefatos não encontrados — gerando agora…")
    tfidf.build_embeddings()
    tfidf.save_embeddings(TFIDF_EMB_PATH)

print("✅ Modelos prontos!\n")

# ── Dash app ────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="EVG — Recomendador de Cursos",
    update_title=None,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Descubra cursos gratuitos da Escola Virtual do Governo com IA."},
    ],
)

server = app.server  # for gunicorn / production


# ── Helper: build course cards ──────────────────────────────────────────
def _make_cards(df: pd.DataFrame) -> list:
    if df.empty:
        return [html.Div("Nenhum curso encontrado acima do limiar.", className="empty-state")]
    cards = []
    for _, row in df.iterrows():
        cards.append(
            html.Div(
                [
                    html.Div(row["nome_curso"], className="course-name"),
                    html.Div(row.get("apresentacao", ""), className="course-desc"),
                    html.Div(
                        [
                            html.Span(f"Similaridade: {row['similaridade']}", className="sim-score"),
                            html.Span(f"{row.get('carga_horaria', '—')}h", className="hours"),
                        ],
                        className="course-meta",
                    ),
                ],
                className="course-card",
            )
        )
    return cards


# ── Layout ──────────────────────────────────────────────────────────────
app.layout = html.Div(
    className="app-container",
    children=[
        # Header
        html.Div(
            className="header",
            children=[
                html.H1("Buscador de cursos EVG"),
                html.P("Descubra cursos gratuitos da Escola Virtual do Governo com auxílio da IA."),
            ],
        ),
        # Input section
        html.Div(
            className="input-section",
            children=[
                html.Label("Descreva sua necessidade de capacitação"),
                dcc.Textarea(
                    id="query-input",
                    placeholder="Ex.: Preciso aprender sobre gestão de projetos e liderança no setor público…",
                    style={"width": "100%"},
                ),
                # Slider
                html.Div(
                    className="slider-section",
                    children=[
                        html.Div("Limiar mínimo de similaridade do cosseno", className="slider-label"),
                        dcc.Slider(
                            id="threshold-slider",
                            min=0,
                            max=1,
                            step=0.1,
                            value=0.0,
                            marks={round(i * 0.1, 1): str(round(i * 0.1, 1)) for i in range(11)},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Button("Buscar Recomendações", id="search-btn", className="search-btn", n_clicks=0),
            ],
        ),
        # Results
        html.Div(id="results-area"),
        # Footer
        html.Div(
            className="footer",
            children=[
                html.Div(
                    className="disclaimer-box",
                    children=[
                        html.Span("⚠️ Este aplicativo não é oficial da EVG. Ele utiliza apenas os dados públicos do "),
                        html.A("Catálogo de Cursos", href="https://www.escolavirtual.gov.br/catalogo", target="_blank"),
                        html.Span(" disponibilizado pela Escola Virtual de Governo."),
                    ]
                ),
                html.Div(
                    children=[
                        html.Span("Atualização semanal automática"),
                    ],
                    style={"marginTop": "16px"}
                )
            ],
        ),
    ],
)


# ── Callback ────────────────────────────────────────────────────────────
@app.callback(
    Output("results-area", "children"),
    Input("search-btn", "n_clicks"),
    State("query-input", "value"),
    State("threshold-slider", "value"),
    prevent_initial_call=True,
)
def search(n_clicks, query, threshold):
    if not query or not query.strip():
        return html.Div("Por favor, descreva sua necessidade de capacitação.", className="empty-state")

    threshold = float(threshold or 0.0)

    sbert_results = sbert.recommend(query, top_n=3, threshold=threshold)
    tfidf_results = tfidf.recommend(query, top_n=3, threshold=threshold)

    return html.Div(
        className="results-grid",
        children=[
            # SBERT panel
            html.Div(
                className="result-panel",
                children=[
                    html.Div(
                        [html.Span("SBERT", className="badge badge-sbert"), " recomenda:"],
                        className="panel-title",
                    ),
                    html.Div(_make_cards(sbert_results)),
                ],
            ),
            # TF-IDF panel
            html.Div(
                className="result-panel",
                children=[
                    html.Div(
                        [html.Span("TF-IDF", className="badge badge-tfidf"), " recomenda:"],
                        className="panel-title",
                    ),
                    html.Div(_make_cards(tfidf_results)),
                ],
            ),
        ],
    )


# ── Run ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
