# Battery Data Pipeline

This project processes battery time series data from the measurements_coding_challenge.csv file, performs data cleansing and transformation operations, and outputs the results to a CSV file.

## Features

- **Data Cleaning**:
  - Handles missing values and invalid data formats (`n/a`, `null` strings)
  - Converts columns to appropriate data types
  - Removes outliers based on statistical analysis
  - Handles corrupt values

- **Data Transformation**:
  - Calculates total grid purchase and feed-in by hour
  - Identifies the hour with highest grid feed-in for each day
  - Adds a boolean flag for the max feed-in hour

- **Containerization**:
  - Complete Docker setup for easy deployment
  - Configurable input/output paths via environment variables

## Project Structure

```
.
├── battery_data_pipeline.py    # Main application code
├── data/                       # Data directory for CSV files
│   └── measurements_coding_challenge.csv  # Input data file
├── Dockerfile                  # Docker configuration
├── README.md                   # This documentation file
├── build_and_run.sh            # Helper script to build and run the container
└── requirements.txt            # Python dependencies
```

## Prerequisites

- Docker
- Git (for cloning the repository)

## Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/battery-data-pipeline.git
   cd battery-data-pipeline
   ```

2. Copy the input CSV file to the data directory:
   ```
   mkdir -p data
   cp /path/to/measurements_coding_challenge.csv data/
   ```

3. Build and run the pipeline using the helper script:
   ```
   chmod +x build_and_run.sh
   ./build_and_run.sh
   ```

4. Find the cleaned data in `data/cleaned_battery_data.csv`

## Manual Docker Commands

If you prefer to run the Docker commands manually:

1. Build the Docker image:
   ```
   docker build -t battery-data-pipeline .
   ```

2. Run the container:
   ```
   docker run --rm -v $(pwd)/data:/app/data battery-data-pipeline
   ```

## Configuration

You can configure the pipeline using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| INPUT_FILE | Path to input CSV file | /app/data/measurements_coding_challenge.csv |
| OUTPUT_FILE | Path to output CSV file | /app/data/cleaned_battery_data.csv |

Example with custom paths:
```
docker run --rm \
  -v $(pwd)/data:/app/data \
  -e INPUT_FILE=/app/data/my_custom_input.csv \
  -e OUTPUT_FILE=/app/data/my_custom_output.csv \
  battery-data-pipeline
```

## Input Data Format

The application expects a semicolon-delimited CSV file with the following columns:
- `timestamp`: Date and time of the measurement (ISO format)
- `serial`: Serial number identifying the battery
- `grid_purchase`: Energy purchased from the grid
- `grid_feedin`: Energy fed back to the grid
- `direct_consumption`: Direct energy consumption (often null in the source data)
- `date`: Date in YYYY-MM-DD format

The input file may contain data quality issues like:
- Missing values represented as empty strings
- Invalid values represented as "n/a" or "null" strings
- Incorrect data types

## Output Data Format

The output CSV will contain all original columns plus:
- `hour_of_day`: Hour extracted from timestamp
- `total_grid_purchase`: Sum of grid purchases across all batteries for that hour
- `total_grid_feedin`: Sum of grid feed-in across all batteries for that hour
- `is_max_feedin_hour`: Boolean flag indicating if this hour had the highest grid feed-in for the day

## Data Quality Handling

The pipeline handles various data quality issues:

1. **Missing Values**:
   - "n/a" in grid_purchase is replaced with 0
   - "null" strings in direct_consumption are replaced with 0
   - Empty cells are replaced with appropriate default values

2. **Data Type Conversion**:
   - Numeric columns are converted to proper numeric types
   - Timestamps are properly parsed to datetime objects

3. **Outlier Detection and Handling**:
   - Uses IQR method to identify statistical outliers
   - Outliers are capped at reasonable bounds rather than removed

## Development

### Running Locally (Without Docker)

1. Set up a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the pipeline:
   ```
   python battery_data_pipeline.py
   ```

## License

This project is licensed under the MIT License.