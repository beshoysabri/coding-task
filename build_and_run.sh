#!/bin/bash

# Exit on error
set -e

echo "========================================="
echo "Battery Data Pipeline - Build and Run"
echo "========================================="

# Create data directory if it doesn't exist
mkdir -p data

# Check if the input file exists in the data directory
if [ ! -f data/measurements_coding_challenge.csv ]; then
    echo "❌ Input file data/measurements_coding_challenge.csv not found!"
    echo "Please copy the measurements_coding_challenge.csv file to the data directory."
    
    # Prevent terminal from closing
    echo ""
    echo "Press Enter to exit..."
    read -r
    exit 1
fi

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t battery-data-pipeline .

# Run the container
echo "🚀 Running pipeline in Docker container..."
docker run --rm -v "$(pwd)/data:/app/data" battery-data-pipeline

# Check if the output file was created
if [ -f data/cleaned_battery_data.csv ]; then
    echo "========================================="
    echo "✅ Pipeline completed successfully!"
    echo "Output file: data/cleaned_battery_data.csv"
    echo "========================================="
    
    # Print the first few lines of the output file
    echo "Preview of output file:"
    head -n 5 data/cleaned_battery_data.csv
else
    echo "❌ Error: Pipeline did not generate an output file."
    
    # Prevent terminal from closing on error
    echo ""
    echo "Press Enter to exit..."
    read -r
    exit 1
fi

# Prevent terminal from closing automatically
echo ""
echo "Pipeline execution completed. Press Enter to exit..."
read -r