# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies
# pandas and numpy are the primary dependencies for this application
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY battery_data_pipeline.py .

# Create data directories
RUN mkdir -p data

# Set environment variables (can be overridden at runtime)
ENV INPUT_FILE=/app/data/measurements_coding_challenge.csv
ENV OUTPUT_FILE=/app/data/cleaned_battery_data.csv

# Run the data pipeline when the container starts
ENTRYPOINT ["python", "battery_data_pipeline.py"]