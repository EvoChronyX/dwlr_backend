# Dockerfile for YOUR Advanced Groundwater Monitoring System
# Based on YOUR groundwater_monitoring_interface.py with YOUR high-accuracy model
FROM python:3.11-slim

WORKDIR /app

# System dependencies for YOUR ensemble model
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for YOUR system
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy YOUR complete monitoring system
COPY complete_groundwater_system.py ./
COPY your_high_accuracy_model.py ./
COPY main.py ./

# Copy training data
COPY train_dataset.csv ./
RUN mkdir -p data && cp train_dataset.csv data/train_dataset.csv || echo "Dataset copy skipped"

# Environment for YOUR system
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Health check for YOUR system
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run YOUR advanced monitoring system
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
