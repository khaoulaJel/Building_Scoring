# Use the official Python 3.13 slim image
FROM python:3.13-slim

# Set the working directory
WORKDIR /app

# Install system deps for postgres client and psycopg2
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Default command
CMD ["python", "main.py"]
