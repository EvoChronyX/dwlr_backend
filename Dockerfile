# Minimal Dockerfile for local testing / alternative deployment
FROM python:3.11-slim

WORKDIR /app

# System deps (if any heavy libs needed, add here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./
RUN mkdir -p data && cp train_dataset.csv data/train_dataset.csv || echo "Dataset copy skipped"

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
