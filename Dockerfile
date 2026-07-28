# ChordMiniApp Python Backend Dockerfile
# Multi-stage build for optimized Flask ML service deployment

# Stage 1: Builder
FROM python:3.10-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies including build tools for ML libraries
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libsndfile1-dev \
    ffmpeg \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install build dependencies first for madmom compilation.
# Pin setuptools to the pre-upgrade version observed in Cloud Build logs so
# pkg_resources remains available for madmom/librosa runtime paths.
RUN pip install --no-cache-dir --upgrade pip "setuptools==79.0.1" wheel
RUN pip install --no-cache-dir Cython>=0.29.0 numpy==1.26.4
RUN pip install --no-cache-dir git+https://github.com/CPJKU/madmom
# Install requirements (Spleeter excluded — TensorFlow too large for Render free tier 512MB)
RUN grep -v '^spleeter==' requirements.txt | grep -v '^typer==' > requirements_nospleeter.txt \
    && pip install --no-cache-dir -r requirements_nospleeter.txt \
    && pip install --no-cache-dir --no-deps typer==0.9.0

# Clone Chord-CNN-LSTM model (Python code + weights) in builder stage
RUN git clone --depth 1 https://github.com/ptnghia-j/chord-cnn-lstm-model.git /tmp/chord-cnn-lstm

# Stage 2: Runtime
FROM python:3.10-slim as runtime

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Sanity-check that the copied runtime environment still exposes the legacy
# pkg_resources module required by madmom.
RUN python -c "import pkg_resources; import madmom; print('madmom runtime import ok', getattr(madmom, '__version__', 'unknown'))"

# Copy application code and essential files
COPY app.py .
COPY app_factory.py .
COPY config.py .
COPY config/ config/
COPY services/ services/
COPY blueprints/ blueprints/
COPY models/ models/
# Copy Chord-CNN-LSTM model from builder (cloned from ptnghia-j/chord-cnn-lstm-model)
COPY --from=builder /tmp/chord-cnn-lstm models/Chord-CNN-LSTM/
COPY utils/ utils/
COPY extensions.py .
COPY error_handlers.py .

COPY compat/ compat/
# Ensure legacy scipy_patch.py is not present (use compat/ patches instead)
RUN rm -f /app/scipy_patch.py || true

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1001 app \
    && chown -R app:app /app

USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Set environment variables
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application with optimized settings for ML processing
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 600 --worker-class sync --max-requests 100 app:app