@echo off
setlocal enabledelayedexpansion

echo =========================================
echo Battery Data Pipeline - Build and Run
echo =========================================

:: Create data directory if it doesn't exist
if not exist data mkdir data

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
) else (
    echo [31m❌ Error: Pipeline did not generate an output file.[0m
    echo.
    echo Press any key to exit...
    pause > nul
    exit /b 1
)

:: Prevent terminal from closing automatically
echo.
echo Pipeline execution completed. Press any key to exit...
pause > nul