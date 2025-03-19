#!/usr/bin/env python3
"""
Battery Data Pipeline

This script processes battery time series data from the measurements_coding_challenge.csv file,
cleans it, performs transformations, and outputs the results to a CSV file.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class BatteryDataPipeline:
    """Data pipeline for processing battery time series data."""
    
    def __init__(self, input_file=None, output_file=None):
        """
        Initialize the data pipeline.
        
        Args:
            input_file (str): Path to the input CSV file
            output_file (str): Path to save the output CSV file
        """
        self.input_file = input_file or os.environ.get('INPUT_FILE', 'data/measurements_coding_challenge.csv')
        self.output_file = output_file or os.environ.get('OUTPUT_FILE', 'data/cleaned_battery_data.csv')
        self.df = None
    
    def load_data(self):
        """Load data from semicolon-delimited CSV file."""
        logger.info(f"Loading data from {self.input_file}")
        try:
            # First, read the file directly as text to examine the raw content
            with open(self.input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                if len(lines) > 0:
                    header_line = lines[0].strip()
                    logger.info(f"File header: {header_line}")
                    
                    # Count columns based on semicolon delimiter
                    column_count = header_line.count(';') + 1
                    logger.info(f"Detected {column_count} columns with semicolon delimiter")
                    
                    # Analyze grid_feedin values directly from file
                    if len(lines) > 1:
                        # Get column positions
                        columns = header_line.split(';')
                        try:
                            grid_feedin_index = columns.index('grid_feedin')
                            logger.info(f"grid_feedin is column index {grid_feedin_index}")
                            
                            # Check the first 10 data lines for non-zero grid_feedin values
                            non_zero_count = 0
                            examples = []
                            
                            for i in range(1, min(11, len(lines))):
                                cells = lines[i].strip().split(';')
                                if len(cells) > grid_feedin_index:
                                    value = cells[grid_feedin_index]
                                    if value != '0':
                                        non_zero_count += 1
                                        if len(examples) < 3:
                                            timestamp = cells[0] if len(cells) > 0 else "unknown"
                                            serial = cells[1] if len(cells) > 1 else "unknown"
                                            examples.append((timestamp, serial, value))
                            
                            logger.info(f"Found {non_zero_count} non-zero grid_feedin values in first 10 data rows")
                            if examples:
                                logger.info(f"Examples of non-zero grid_feedin values from raw file: {examples}")
                        except ValueError:
                            logger.error("Could not find grid_feedin column in header")
                    
                    # Show first data line
                    if len(lines) > 1:
                        first_data_line = lines[1].strip()
                        logger.info(f"First data line: {first_data_line}")
                else:
                    logger.error("File appears to be empty")
            
            # Now read the CSV with pandas, specifying correct delimiter and not converting strings yet
            try:
                logger.info("Reading CSV file with pandas")
                self.df = pd.read_csv(
                    self.input_file,
                    sep=';',
                    dtype=str,  # Keep everything as strings initially
                    encoding='utf-8',
                    on_bad_lines='warn'    # Warn about bad lines but don't fail
                )
                logger.info(f"Successfully read CSV with pandas, got {len(self.df)} rows")
            except Exception as e:
                logger.error(f"Error reading CSV with pandas: {str(e)}")
                # Fallback method: read manually line by line
                logger.info("Trying fallback method for reading CSV")
                try:
                    with open(self.input_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Parse header
                    header = lines[0].strip().split(';')
                    
                    # Parse data rows
                    data = []
                    for i in range(1, len(lines)):
                        row = lines[i].strip().split(';')
                        # Ensure row has same length as header
                        if len(row) == len(header):
                            data.append(row)
                    
                    # Create DataFrame
                    self.df = pd.DataFrame(data, columns=header)
                    logger.info(f"Successfully read CSV using fallback method, got {len(self.df)} rows")
                except Exception as e2:
                    logger.error(f"Fallback method also failed: {str(e2)}")
                    return False
            
            # Log column names and some basic stats
            logger.info(f"Loaded columns: {', '.join(self.df.columns)}")
            logger.info(f"Loaded {len(self.df)} rows")
            
            # Check for non-zero grid_feedin values in the raw data
            non_zero_feedin_count = sum(1 for val in self.df['grid_feedin'] if val != '0')
            logger.info(f"Found {non_zero_feedin_count} non-zero grid_feedin values in raw data")
            
            if non_zero_feedin_count > 0:
                # Log some examples
                sample_non_zero = [
                    (row['timestamp'], row['serial'], row['grid_feedin']) 
                    for _, row in self.df.iterrows() 
                    if row['grid_feedin'] != '0'
                ][:5]
                logger.info(f"Sample non-zero grid_feedin entries: {sample_non_zero}")
            
            # Convert timestamp to datetime
            try:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                logger.info("Successfully converted timestamp to datetime")
            except Exception as e:
                logger.error(f"Error converting timestamp to datetime: {str(e)}")
            
            # Also convert date to datetime if needed
            try:
                self.df['date'] = pd.to_datetime(self.df['date'])
                logger.info("Successfully converted date to datetime")
            except Exception as e:
                logger.error(f"Error converting date to datetime: {str(e)}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def clean_data(self):
        """Clean the data by handling missing values, incorrect types, and duplicates."""
        if self.df is None or self.df.empty:
            logger.error("No data loaded to clean")
            return False
            
        logger.info("Starting data cleaning process")
        
        # Store original row count for comparison
        original_row_count = len(self.df)
        
        # Log the data types and some sample values before cleaning
        logger.info("Data types before cleaning:")
        for col in self.df.columns:
            filtered_col = self.df[col].dropna()
            if len(filtered_col) > 0:
                # Only try to sample if there are values available
                if len(filtered_col) >= 3:
                    sample_vals = filtered_col.sample(3).tolist()
                else:
                    # If fewer than 3 values, take all available
                    sample_vals = filtered_col.tolist()
                logger.info(f"  {col} (type: {self.df[col].dtype}): sample values = {sample_vals}"
                           f" (non-null count: {len(filtered_col)})")
            else:
                logger.info(f"  {col} (type: {self.df[col].dtype}): no non-null values")
        
        # 1. Handle grid_purchase column with 'n/a' values
        if 'grid_purchase' in self.df.columns:
            logger.info("Cleaning grid_purchase column")
            
            # Count n/a values before replacement
            na_count = (self.df['grid_purchase'] == 'n/a').sum()
            logger.info(f"Found {na_count} 'n/a' values in grid_purchase column")
            
            # Replace 'n/a' with NaN for proper handling
            self.df['grid_purchase'] = self.df['grid_purchase'].replace('n/a', np.nan)
            
            # Convert grid_purchase to numeric (float)
            # Use errors='coerce' to convert any non-numeric values to NaN
            self.df['grid_purchase'] = pd.to_numeric(self.df['grid_purchase'], errors='coerce')
            
            # Fill NaN values with 0
            nan_count = self.df['grid_purchase'].isna().sum()
            logger.info(f"Found {nan_count} NaN values in grid_purchase after conversion")
            self.df['grid_purchase'] = self.df['grid_purchase'].fillna(0)
        
        # 2. Handle grid_feedin column
        if 'grid_feedin' in self.df.columns:
            logger.info("Cleaning grid_feedin column")
            
            # Log non-zero values before cleaning
            non_zero_before = (self.df['grid_feedin'] != '0').sum()
            logger.info(f"Found {non_zero_before} non-zero grid_feedin values before cleaning")
            
            # Sample some non-zero values before conversion
            if non_zero_before > 0:
                try:
                    # Try to get up to 3 non-zero examples
                    sample_rows = self.df[self.df['grid_feedin'] != '0'].head(3)
                    logger.info(f"Found {len(sample_rows)} non-zero grid_feedin examples to display")
                    for _, row in sample_rows.iterrows():
                        logger.info(f"  Non-zero grid_feedin example: {row['timestamp']} - {row['serial']} - {row['grid_feedin']}")
                except Exception as e:
                    logger.error(f"Error sampling non-zero grid_feedin values: {str(e)}")
            
            # Convert grid_feedin to numeric (float)
            # Use errors='coerce' to convert any non-numeric values to NaN
            self.df['grid_feedin'] = pd.to_numeric(self.df['grid_feedin'], errors='coerce')
            
            # Fill NaN values with 0
            nan_count = self.df['grid_feedin'].isna().sum()
            logger.info(f"Found {nan_count} NaN values in grid_feedin after conversion")
            self.df['grid_feedin'] = self.df['grid_feedin'].fillna(0)
            
            # Log non-zero values after cleaning
            non_zero_after = (self.df['grid_feedin'] > 0).sum()
            logger.info(f"Found {non_zero_after} non-zero grid_feedin values after cleaning")
            
            # Verify some non-zero values are still present
            if non_zero_after > 0:
                sample_rows = self.df[self.df['grid_feedin'] > 0].head(3)
                for _, row in sample_rows.iterrows():
                    logger.info(f"  Non-zero grid_feedin after cleaning: {row['timestamp']} - {row['serial']} - {row['grid_feedin']}")
        
        # 3. Handle direct_consumption column
        if 'direct_consumption' in self.df.columns:
            logger.info("Cleaning direct_consumption column")
            
            # Replace 'null' string with NaN
            null_count = (self.df['direct_consumption'] == 'null').sum()
            logger.info(f"Found {null_count} 'null' strings in direct_consumption")
            self.df['direct_consumption'] = self.df['direct_consumption'].replace('null', np.nan)
            
            # Convert to numeric
            self.df['direct_consumption'] = pd.to_numeric(self.df['direct_consumption'], errors='coerce')
            
            # Fill NaN values with 0
            nan_count = self.df['direct_consumption'].isna().sum()
            logger.info(f"Found {nan_count} NaN values in direct_consumption after conversion")
            self.df['direct_consumption'] = self.df['direct_consumption'].fillna(0)
        
        # 4. Handle serial column - ensure it's treated as a string
        if 'serial' in self.df.columns:
            self.df['serial'] = self.df['serial'].astype(str)
        
        # 5. Remove duplicate rows (timestamp + serial combinations should be unique)
        if 'timestamp' in self.df.columns and 'serial' in self.df.columns:
            logger.info("Checking for duplicate timestamp + serial combinations")
            duplicates = self.df.duplicated(subset=['timestamp', 'serial'], keep='first')
            duplicate_count = duplicates.sum()
            
            if duplicate_count > 0:
                logger.info(f"Removing {duplicate_count} duplicate rows")
                self.df = self.df[~duplicates]
        
        # 6. Handle extreme outliers but avoid removing legitimate values
        for column in ['grid_purchase', 'grid_feedin']:
            if column in self.df.columns:
                # Check if there are non-zero values to analyze
                if (self.df[column] > 0).sum() > 0:
                    # Calculate statistics only on positive values to avoid skewing by zeros
                    positive_values = self.df[self.df[column] > 0][column]
                    
                    # Calculate IQR for determining outliers
                    Q1 = positive_values.quantile(0.25)
                    Q3 = positive_values.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define bounds for outliers (3 IQRs from Q1/Q3)
                    # Set a minimum lower bound of 0 to avoid negative values
                    lower_bound = max(0, Q1 - 3 * IQR)
                    upper_bound = Q3 + 3 * IQR
                    
                    logger.info(f"Outlier bounds for {column}: lower={lower_bound}, upper={upper_bound}")
                    
                    # Count outliers
                    outliers = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).sum()
                    
                    if outliers > 0:
                        logger.info(f"Found {outliers} outliers in {column}")
                        
                        # Clip values to reasonable bounds
                        self.df[column] = self.df[column].clip(lower_bound, upper_bound)
        
        # Log data summary after cleaning
        logger.info("Data summary after cleaning:")
        for col in ['grid_purchase', 'grid_feedin']:
            if col in self.df.columns:
                non_zero = (self.df[col] > 0).sum()
                max_val = self.df[col].max()
                mean_val = self.df[col].mean()
                logger.info(f"  {col}: non-zero values = {non_zero}, max = {max_val}, mean = {mean_val:.2f}")
        
        # Report cleaning results
        cleaned_row_count = len(self.df)
        removed_rows = original_row_count - cleaned_row_count
        logger.info(f"Data cleaning completed. Removed {removed_rows} rows ({removed_rows/original_row_count:.2%})")
        
        return True
    
    def transform_data(self):
        """Perform required transformations on the data."""
        if self.df is None or self.df.empty:
            logger.error("No data loaded to transform")
            return False
            
        logger.info("Starting data transformation")
        
        # Ensure timestamp column exists and is datetime type
        if 'timestamp' not in self.df.columns:
            logger.error("Required 'timestamp' column not found")
            return False
        
        # 1. Extract hour of day
        logger.info("Adding hour of day column")
        self.df['hour_of_day'] = self.df['timestamp'].dt.hour
        
        # 2. Calculate total grid_purchase and grid_feedin by hour
        if 'grid_purchase' in self.df.columns and 'grid_feedin' in self.df.columns:
            logger.info("Calculating total grid metrics by hour")
            
            # Verify grid_feedin values before aggregation
            non_zero_feedin = (self.df['grid_feedin'] > 0).sum()
            logger.info(f"Before hourly calculation: {non_zero_feedin} non-zero grid_feedin values")
            
            # Ensure date is in datetime format for grouping
            if not pd.api.types.is_datetime64_dtype(self.df['date']):
                logger.info("Converting date to datetime format")
                self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Create a copy of the original values before grouping
            self.df['original_grid_feedin'] = self.df['grid_feedin']
            
            # Log distribution of hour values
            hour_counts = self.df['hour_of_day'].value_counts().sort_index()
            logger.info(f"Hour distribution: {hour_counts.to_dict()}")
            
            # Group by date and hour to get hourly totals for all batteries
            logger.info("Aggregating grid metrics by hour")
            hourly_totals = self.df.groupby(['date', 'hour_of_day']).agg({
                'grid_purchase': 'sum',
                'grid_feedin': 'sum'
            }).reset_index()
            
            # Log hourly totals information
            non_zero_hourly_feedin = (hourly_totals['grid_feedin'] > 0).sum()
            logger.info(f"Hourly aggregation: {non_zero_hourly_feedin} hours with non-zero grid_feedin")
            if non_zero_hourly_feedin > 0:
                max_hour = hourly_totals.loc[hourly_totals['grid_feedin'].idxmax()]
                logger.info(f"Max hourly grid_feedin: {max_hour['grid_feedin']} at hour {max_hour['hour_of_day']}")
            
            # Rename columns to indicate these are totals
            hourly_totals.rename(columns={
                'grid_purchase': 'total_grid_purchase',
                'grid_feedin': 'total_grid_feedin'
            }, inplace=True)
            
            # Merge the hourly totals back to the main dataframe
            logger.info("Merging hourly totals back to main dataframe")
            self.df = pd.merge(
                self.df, 
                hourly_totals,
                on=['date', 'hour_of_day'],
                how='left'
            )
            
            # 3. Add a column indicating hour with highest grid_feedin for each day
            logger.info("Identifying hour with highest grid_feedin for each day")
            
            # Find hour with max grid_feedin for each day
            try:
                if non_zero_hourly_feedin > 0:
                    # Create a function to handle finding the max hour safely
                    def get_max_feedin_hour(group):
                        if group['total_grid_feedin'].max() > 0:
                            return group.loc[group['total_grid_feedin'].idxmax()]['hour_of_day']
                        else:
                            return group['hour_of_day'].iloc[0]  # Default to first hour if all are zero
                    
                    max_feedin_hours = hourly_totals.groupby('date').apply(get_max_feedin_hour).reset_index()
                    max_feedin_hours.columns = ['date', 'max_feedin_hour']
                    
                    # Merge max feedin hour information back to main dataframe
                    self.df = pd.merge(self.df, max_feedin_hours, on='date', how='left')
                    
                    # Create boolean column indicating if current hour is max feedin hour
                    self.df['is_max_feedin_hour'] = self.df['hour_of_day'] == self.df['max_feedin_hour']
                    
                    # Drop intermediate column
                    self.df.drop('max_feedin_hour', axis=1, inplace=True)
                else:
                    logger.warning("No non-zero grid_feedin values found, skipping max hour identification")
                    self.df['is_max_feedin_hour'] = False
            except Exception as e:
                logger.error(f"Error identifying max feedin hour: {str(e)}")
                self.df['is_max_feedin_hour'] = False
        else:
            logger.warning("Could not perform grid metrics calculations: required columns missing")
        
        # Verify final data
        logger.info("Verification of output values:")
        for column in ['grid_purchase', 'grid_feedin', 'total_grid_purchase', 'total_grid_feedin']:
            if column in self.df.columns:
                non_zero = (self.df[column] > 0).sum()
                logger.info(f"  {column} has {non_zero} non-zero values")
                if non_zero > 0:
                    logger.info(f"  {column} max value: {self.df[column].max()}")
        
        # Compare original grid_feedin with final values
        if 'original_grid_feedin' in self.df.columns and 'grid_feedin' in self.df.columns:
            orig_non_zero = (self.df['original_grid_feedin'] > 0).sum()
            final_non_zero = (self.df['grid_feedin'] > 0).sum()
            logger.info(f"Original non-zero grid_feedin values: {orig_non_zero}")
            logger.info(f"Final non-zero grid_feedin values: {final_non_zero}")
            
            # Remove the temporary column
            self.df.drop('original_grid_feedin', axis=1, inplace=True)
        
        logger.info("Data transformation completed")
        return True
    
    def save_data(self):
        """Save the processed data to CSV."""
        if self.df is None or self.df.empty:
            logger.error("No data to save")
            return False
            
        logger.info(f"Saving processed data to {self.output_file}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Check one more time for non-zero grid_feedin values
        non_zero_feedin = (self.df['grid_feedin'] > 0).sum()
        logger.info(f"Saving {non_zero_feedin} rows with non-zero grid_feedin values")
        
        # Save to CSV
        self.df.to_csv(self.output_file, index=False)
        logger.info(f"Successfully saved {len(self.df)} rows to {self.output_file}")
        return True
    
    def run_pipeline(self):
        """Run the complete data pipeline."""
        logger.info("Starting battery data pipeline")
        
        # Execute pipeline steps
        if not self.load_data():
            return False
        
        if not self.clean_data():
            return False
        
        if not self.transform_data():
            return False
        
        if not self.save_data():
            return False
        
        logger.info("Battery data pipeline completed successfully")
        return True


if __name__ == "__main__":
    # Initialize and run the pipeline
    pipeline = BatteryDataPipeline()
    success = pipeline.run_pipeline()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)