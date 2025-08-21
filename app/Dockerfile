# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.1.105-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git wget curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies (including GPU-enabled torch >=2.6.0)
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY . .

# Expose Flask port
EXPOSE 8001

# Run Flask app
CMD ["python3", "main.py"]
