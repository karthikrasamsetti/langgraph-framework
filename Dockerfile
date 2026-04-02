FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# libgomp1 is required by some ML packages on Linux
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv — same package manager you use locally
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy requirements first so this layer is cached
# Docker only re-runs this if requirements.txt changes
COPY requirements.txt .

# Install all Python packages
RUN uv pip install --system -r requirements.txt

# Copy all application code
COPY . .

# Pre-download the embedding model into the image
# This means the model is ready at startup — no 40 second delay on first request
# TRANSFORMERS_OFFLINE=0 here so it CAN download, then we set it back to 1 at runtime
RUN python -c "\
import os; \
os.makedirs('./models_cache', exist_ok=True); \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models_cache'); \
print('Embedding model cached successfully')"

# Create directories that will be used as volume mount points
RUN mkdir -p /app/vector_db /app/my_docs /app/logs

# Expose the API port
EXPOSE 8000

# Health check — Docker will monitor this
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the server
CMD ["bash", "scripts/start.sh"]