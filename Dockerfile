FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (curl for healthchecks, build-essential if wheels need compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-api.txt

# Copy application code and verified deployment artifacts
COPY api/ api/
COPY deployment/ deployment/
COPY models/ models/
COPY artifacts/ artifacts/
COPY reports/ reports/
COPY config.yaml .

# Create logs directory
RUN mkdir -p logs/inference

# Expose FastAPI service port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch ASGI production server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
