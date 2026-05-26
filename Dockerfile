# Use official Python slim image
FROM python:3.10-slim

# Install system dependencies for OpenCV and Tesseract OCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR models (English + Tagalog) during build to avoid slow startup
RUN python -c "import easyocr; reader = easyocr.Reader(['en', 'tl'])"

# Copy the rest of the code
COPY . .

# Cloud Run requires listening on the PORT environment variable
ENV PORT 8080

# Run the web service on container startup.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
