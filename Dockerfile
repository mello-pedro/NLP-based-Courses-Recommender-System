FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NLTK stopwords
RUN python -c "import nltk; nltk.download('stopwords', quiet=True)"

# App source
COPY . .

EXPOSE 8050

CMD ["python", "app.py"]
