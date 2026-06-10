# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5001

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (needed for certain C++ compiles or general utilities)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt to container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directories for caching and uploads (matching app config /tmp/docs and /tmp/cache)
RUN mkdir -p /tmp/docs /tmp/cache

# Expose the application port
EXPOSE 5001

# Run the application using Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
