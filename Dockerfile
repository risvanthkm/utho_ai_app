FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8001

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY db ./db
COPY models ./models
COPY data ./data
COPY scripts ./scripts
COPY datasets/raw/inventory.csv ./datasets/raw/inventory.csv
COPY datasets/raw/pricing.csv ./datasets/raw/pricing.csv

EXPOSE 8001

HEALTHCHECK --interval=5s --timeout=5s --start-period=45s --retries=10 \
    CMD python -c "import os, urllib.request; port = os.getenv('PORT', '8001'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=4)" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
