# Battery Data Pipeline

A comprehensive data processing and visualization pipeline for battery time series data. This project analyzes battery metrics, cleans the data, generates visualizations, and creates a PDF report with insights.

## Overview

This pipeline processes battery time series data from CSV files, performing cleaning operations, transformations, and visualizations. The pipeline handles common data quality issues such as missing values, incorrect data types, and inconsistent formatting. The processed data is visualized through multiple chart types and compiled into a professional PDF report.

## File Structure

```
.
├── analysis/                        # Generated visualizations and report
│   ├── battery_comparison.png       # Battery performance comparison chart
│   ├── battery_data_report.pdf      # Generated PDF report
│   ├── feedin_proportion.png        # Pie chart of grid feedin distribution
│   ├── grid_activity_heatmap.png    # Heatmap of grid activity
│   ├── grid_metrics_distribution.png # Distribution of grid metrics
│   ├── grid_metrics_time_series.png # Time series of grid metrics
│   ├── hourly_grid_activity.png     # Hourly grid activity chart
│   └── total_grid_metrics_by_hour.png # Hourly grid metrics
├── data/                            # Data files
│   ├── cleaned_battery_data.csv     # Output from the pipeline
│   └── measurements_coding_challenge.csv # Input data file
├── battery_data_pipeline.py         # Main data processing script
├── build_and_run.bat                # Windows script (pipeline only)
├── build_and_run.sh                 # Linux/macOS script (pipeline only)
├── build_and_run_charts.bat         # Windows script (pipeline + charts)
├── build_and_run_charts.sh          # Linux/macOS script (pipeline + charts)
├── build_and_run_charts_report.bat  # Windows script (full pipeline with PDF)
├── build_and_run_charts_report.sh   # Linux/macOS script (full pipeline with PDF)
├── Dockerfile                       # Docker configuration
├── generate_charts.py               # Visualization generation script
├── generate_pdf_report.py           # PDF report generation script
├── README.md                        # This documentation file
└── requirements.txt                 # Python dependencies
```

## Prerequisites

- Docker
- Git (for cloning the repository)

## Quick Start

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/battery-data-pipeline.git
cd battery-data-pipeline
```

### Step 2: Prepare the input data

Copy your input CSV file to the `data` directory:

```bash
mkdir -p data
cp /path/to/measurements_coding_challenge.csv data/
```

### Step 3: Run the pipeline

Choose one of the following scripts based on your needs:

#### Basic Pipeline (Data Cleaning Only)

**Linux/macOS:**
```bash
chmod +x build_and_run.sh
./build_and_run.sh
```

**Windows:**
```
build_and_run.bat
```

#### Pipeline with Charts

**Linux/macOS:**
```bash
chmod +x build_and_run_charts.sh
./build_and_run_charts.sh
```

**Windows:**
```
build_and_run_charts.bat
```

#### Complete Pipeline with Charts and PDF Report

**Linux/macOS:**
```bash
chmod +x build_and_run_charts_report.sh
./build_and_run_charts_report.sh
```

**Windows:**
```
build_and_run_charts_report.bat
```

## Execution Options

This pipeline offers multiple execution options to fit different needs:

1. **Basic Data Pipeline** (`build_and_run.*`)
   - Cleans and processes the raw battery data
   - Outputs `cleaned_battery_data.csv`

2. **Data Pipeline with Visualizations** (`build_and_run_charts.*`)
   - Runs the data pipeline
   - Generates visualizations in the `analysis` directory

3. **Complete Pipeline with PDF Report** (`build_and_run_charts_report.*`)
   - Runs the data pipeline
   - Generates visualizations
   - Creates a comprehensive PDF report

## Input Data Format

The application expects a semicolon-delimited CSV file with the following columns:
- `timestamp`: Date and time of the measurement (ISO format)
- `serial`: Serial number identifying the battery
- `grid_purchase`: Energy purchased from the grid
- `grid_feedin`: Energy fed back to the grid
- `direct_consumption`: Direct energy consumption
- `date`: Date in YYYY-MM-DD format

## Output Files

### Cleaned Data
The pipeline produces a cleaned CSV file (`data/cleaned_battery_data.csv`) with:
- Properly formatted timestamps
- Correct data types
- No missing values
- Additional calculated metrics

### Visualizations
The following charts are generated in the `analysis` directory:
- **Hourly Grid Activity**: Line chart showing average grid purchase and feed-in by hour
- **Total Grid Metrics by Hour**: Bar chart of total grid metrics across all batteries
- **Grid Metrics Distribution**: Histograms of grid purchase and feed-in distributions
- **Battery Comparison**: Bar chart comparing batteries' grid interactions
- **Grid Activity Heatmap**: Heatmap showing activity patterns by hour and battery
- **Grid Metrics Time Series**: Time series of grid metrics over time
- **Feed-in Proportion**: Pie chart showing each battery's contribution to grid feed-in

**NOTE:** Having only one day data in the given data table will render The Time Series chart as well the Total Grid Metrics Chart useless.
         The Total Grid Metrics chart is intended to view the averaging of total metrics for multiple days, having one day in this case is useless.
         Also the Timeseries becomes useless because it will only show the data for the single day ( same as the first chart )

### PDF Report
A comprehensive PDF report (`analysis/battery_data_report.pdf`) with:
- Executive summary of findings
- Statistical analysis
- All visualizations with explanations
- Methodology and conclusions

## Docker Details

All processing occurs in Docker containers to ensure reproducibility:
- The main pipeline uses `battery-data-pipeline` image
- Visualization and reporting use `battery-data-viz` image ( Installed on-the-fly by the bash scripts)
- All dependencies are installed automatically

## Development

### Custom Processing

To customize the data processing, modify `battery_data_pipeline.py`.

### Custom Visualizations

To add or modify visualizations, edit `generate_charts.py`.

### Custom Reporting

To change the PDF report structure or content, modify `generate_pdf_report.py`.

## Troubleshooting

- **Missing input file**: Ensure `measurements_coding_challenge.csv` is in the `data` directory
- **Docker errors**: Make sure Docker is installed and running
- **Permission issues**: Ensure the bash scripts have execution permissions (`chmod +x *.sh`)
- **Write Files Permissions issues**: Please Make Sure the Resulting files are closed on the OS before rerunning the scripts.

## License

This project is licensed under the MIT License.