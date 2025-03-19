#!/bin/bash

# Exit on error
set -e

echo "========================================="
echo "Battery Data Pipeline - Build and Run"
echo "========================================="

# Create data directory if it doesn't exist
mkdir -p data
# Create analysis directory for visualization output
mkdir -p analysis

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

    # Run visualization script
    echo ""
    echo "🎨 Generating visualization charts..."
    
    # Create a temporary Docker image for visualization
    echo "Building visualization image..."
    docker build -t battery-data-viz - <<EOF
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate_charts.py .

ENTRYPOINT ["python", "generate_charts.py"]
EOF

    # Run the visualization container
    docker run --rm \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/analysis:/app/analysis" \
        -v "$(pwd)/generate_charts.py:/app/generate_charts.py" \
        -e INPUT_FILE=/app/data/cleaned_battery_data.csv \
        -e OUTPUT_DIR=/app/analysis \
        battery-data-viz

    # Check if charts were created
    CHART_COUNT=$(find analysis -name "*.png" | wc -l)
    if [ "$CHART_COUNT" -gt 0 ]; then
        echo "✅ Generated $CHART_COUNT visualization charts in the 'analysis' folder!"
        # List the generated charts
        echo "Generated charts:"
        ls -1 analysis/*.png
    else
        echo "❌ No visualization charts were generated."
    fi
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
echo "Pipeline and visualization completed. Press Enter to exit..."
read -r