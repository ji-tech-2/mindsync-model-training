FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY custom_ridge.py .
COPY train.py .

# Copy dataset
COPY df/ ./df/

# Create artifacts directory
RUN mkdir -p artifacts

# Environment variables (override these at runtime)
ENV WANDB_PROJECT=mindsync-model
ENV MODEL_VERSION=v1.0
ENV PYTHONUNBUFFERED=1

# Run training script
CMD ["python", "train.py"]
