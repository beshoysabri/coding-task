@echo off
setlocal enabledelayedexpansion

echo =========================================
echo Battery Data Pipeline - Build and Run
echo =========================================

:: Create data directory if it doesn't exist
if not exist data mkdir data
:: Create analysis directory for visualization output
if not exist analysis mkdir analysis

:: Check if the input file exists in the data directory
if not exist data\measurements_coding_challenge.csv (
    echo [31m❌ Input file data\measurements_coding_challenge.csv not found![0m
    echo Please copy the measurements_coding_challenge.csv file to the data directory.
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

:: Build Docker image
echo [33m🔨 Building Docker image...[0m
docker build -t battery-data-pipeline .
if %ERRORLEVEL% neq 0 (
    echo [31m❌ Docker build failed![0m
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

:: Run the container
echo [33m🚀 Running pipeline in Docker container...[0m
docker run --rm -v "%cd%\data:/app/data" battery-data-pipeline
if %ERRORLEVEL% neq 0 (
    echo [31m❌ Docker run failed![0m
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

:: Check if the output file was created
if exist data\cleaned_battery_data.csv (
    echo =========================================
    echo [32m✅ Pipeline completed successfully![0m
    echo Output file: data\cleaned_battery_data.csv
    echo =========================================
    
    :: Print the first few lines of the output file
    echo Preview of output file:
    type data\cleaned_battery_data.csv | findstr /n . | findstr "^[1-5]:"

    :: Run visualization script
    echo.
    echo [33m🎨 Generating visualization charts...[0m
    
    :: Create a temporary Docker image for visualization
    echo Building visualization image...
    
    :: Write a temporary Dockerfile for visualization
    echo FROM python:3.9-slim > Dockerfile.viz
    echo WORKDIR /app >> Dockerfile.viz
    echo COPY requirements.txt . >> Dockerfile.viz
    echo RUN pip install --no-cache-dir -r requirements.txt >> Dockerfile.viz
    echo COPY generate_charts.py . >> Dockerfile.viz
    echo COPY generate_pdf_report.py . >> Dockerfile.viz
    echo ENTRYPOINT ["python", "generate_charts.py"] >> Dockerfile.viz
    
    :: Build the visualization image
    docker build -t battery-data-viz -f Dockerfile.viz .
    if %ERRORLEVEL% neq 0 (
        echo [31m❌ Visualization build failed![0m
        del Dockerfile.viz
        echo.
        echo Press any key to exit...
        pause > nul
        exit /b 1
    )
    
    :: Remove temporary Dockerfile
    del Dockerfile.viz
    
    :: Run the visualization container
    docker run --rm ^
        -v "%cd%\data:/app/data" ^
        -v "%cd%\analysis:/app/analysis" ^
        -v "%cd%\generate_charts.py:/app/generate_charts.py" ^
        -e INPUT_FILE=/app/data/cleaned_battery_data.csv ^
        -e OUTPUT_DIR=/app/analysis ^
        battery-data-viz
    
    :: Check if charts were created
    set chart_count=0
    for %%F in (analysis\*.png) do (
        set /a chart_count+=1
    )
    
    if !chart_count! gtr 0 (
        echo [32m✅ Generated !chart_count! visualization charts in the 'analysis' folder![0m
        
        :: Generate PDF report
        echo.
        echo [33m📄 Generating PDF report from charts...[0m
        
        :: Run the PDF report generator
        docker run --rm ^
            -v "%cd%\data:/app/data" ^
            -v "%cd%\analysis:/app/analysis" ^
            -v "%cd%\generate_pdf_report.py:/app/generate_pdf_report.py" ^
            -e DATA_FILE=/app/data/cleaned_battery_data.csv ^
            -e CHARTS_DIR=/app/analysis ^
            -e REPORT_FILE=/app/analysis/battery_data_report.pdf ^
            --entrypoint python ^
            battery-data-viz ^
            generate_pdf_report.py
        
        :: Check if the PDF was created
        if exist analysis\battery_data_report.pdf (
            echo [32m✅ Generated PDF report: analysis\battery_data_report.pdf[0m
            echo [33m📝 PDF report contains all visualization charts with explanations.[0m
        ) else (
            echo [31m❌ Failed to generate PDF report.[0m
        )
        
        :: List the generated files
        echo.
        echo Generated output files:
        echo - data\cleaned_battery_data.csv (Cleaned data^)
        for %%F in (analysis\*.png) do (
            echo - %%F (Chart^)
        )
        if exist analysis\battery_data_report.pdf (
            echo - analysis\battery_data_report.pdf (PDF Report^)
        )
    ) else (
        echo [31m❌ No visualization charts were generated.[0m
    )
) else (
    echo [31m❌ Error: Pipeline did not generate an output file.[0m
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

:: Prevent terminal from closing automatically
echo.
echo Pipeline, visualization, and reporting completed. Press any key to exit...
pause > nul