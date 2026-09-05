# Simple Python image - no build tools needed (MediaPipe is pure pip)
FROM python:3.11-slim

# Only need libGL for opencv headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages (all pure pip - no C++ compilation!)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .
RUN mkdir -p uploads

# Railway sets $PORT automatically
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1 --threads 4
