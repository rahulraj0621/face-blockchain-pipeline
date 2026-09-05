# Use official Python slim image
FROM python:3.11-slim

# Install system build tools needed for dlib (C++ face detection)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
# Install dlib first (needs cmake, takes ~5 mins to compile)
COPY requirements.txt .
RUN pip install --no-cache-dir dlib==19.24.2
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the project
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port (Railway sets $PORT automatically)
EXPOSE 8080

# Start gunicorn
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 4
