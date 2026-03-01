#!/usr/bin/env bash
# ── Setup virtual environment for the EVG Course Recommender ──────────
set -e

VENV_DIR=".venv"

echo "🔧 Criando ambiente virtual em ${VENV_DIR}/ …"
python3 -m venv "${VENV_DIR}"

echo "🔧 Ativando ambiente virtual …"
source "${VENV_DIR}/bin/activate"

echo "📦 Instalando dependências …"
pip install --upgrade pip
pip install -r requirements.txt

echo "📥 Baixando stopwords do NLTK …"
python -c "import nltk; nltk.download('stopwords', quiet=True)"

echo ""
echo "✅ Ambiente pronto! Para ativar:"
echo "   source ${VENV_DIR}/bin/activate"
echo ""
echo "Para gerar os embeddings:"
echo "   python scripts/build_embeddings.py"
echo ""
echo "Para rodar o app:"
echo "   python app.py"
