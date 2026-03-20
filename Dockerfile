FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir \
    -r requirements.txt

# Copy all app files
COPY . .

# Create temp directory
RUN mkdir -p /tmp/ai_slop_pipeline

# Expose port (Railway injects $PORT at runtime)
EXPOSE 7860

# Launch
CMD ["python", "app.py"]