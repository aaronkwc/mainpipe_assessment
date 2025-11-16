FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/reports

# Pre-install a small spaCy model used for tests to avoid large runtime downloads.
# Use a specific model version compatible with spaCy 3.x (adjust if you change spaCy version).
RUN pip install --no-cache-dir en-core-web-sm==3.8.0 || true

# Ensure a Python kernel is available for nbconvert ExecutePreprocessor
# This registers a 'python3' kernel that nbconvert will find when executing notebooks.
RUN python -m ipykernel install --sys-prefix --name python3 --display-name python3 || true

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["mainpipe", "--help"]
