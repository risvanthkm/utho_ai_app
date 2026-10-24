FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY datasets/raw/inventory.csv ./datasets/raw/inventory.csv
COPY datasets/raw/pricing.csv ./datasets/raw/pricing.csv

RUN mkdir -p /app/models/tft

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
