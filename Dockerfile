# Lightweight API image that proxies requests to a local Ollama-hosted DeepSeek-OCR model.
FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow/PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libpng-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn[standard]==0.27.0 \
    python-multipart==0.0.6 \
    requests==2.31.0 \
    Pillow==10.2.0 \
    PyMuPDF==1.23.26 \
    tqdm==4.66.1

# Copy application code
COPY start_server.py .

EXPOSE 8000

CMD ["uvicorn", "start_server:app", "--host", "0.0.0.0", "--port", "8000"]
