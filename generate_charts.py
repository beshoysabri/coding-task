#!/usr/bin/env python3
"""
Battery Data Visualization

This script generates visualization charts based on the cleaned battery data
and saves them as PNG files in the analysis folder.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class BatteryDataVisualizer:
    """Generates visualizations from battery data."""
    
    def __init__(self, input_file=None, output_dir=None):
        """
        Initialize the visualizer.
        
        Args:
            input_file (str): Path to the cleaned CSV file
            output_dir (str): Directory to save visualization files
        """
        self.input_file = input_file or os.environ.get('INPUT_FILE', 'data/cleaned_battery_data.csv')
        self.output_dir = output_dir or os.environ.get('OUTPUT_DIR', 'analysis')
        self.df = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plot styling
        plt.style.use('ggplot')
        self.colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    def load_data(self):
        """Load data from the cleaned CSV file."""
        logger.info(f"Loading data from {self.input_file}")
        try:
            self.df = pd.read_csv(self.input_file, parse_dates=['timestamp', 'date'])
            logger.info(f"Successfully loaded {len(self.df)} rows and {len(self.df.columns)} columns")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def save_figure(self, fig, filename):
        """Save a matplotlib figure to a file."""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved chart to {filepath}")
    
    def plot_hourly_grid_activity(self):
        """Create a line chart of average grid activity by hour of day."""
        logger.info("Generating hourly grid activity chart")
        
        # Group by hour and calculate averages
        hourly_avg = self.df.groupby('hour_of_day').agg({
            'grid_purchase': 'mean',
            'grid_feedin': 'mean'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot both metrics
        ax.plot(hourly_avg['hour_of_day'], hourly_avg['grid_purchase'], 
                marker='o', linewidth=2, label='Grid Purchase', color=self.colors[0])
        ax.plot(hourly_avg['hour_of_day'], hourly_avg['grid_feedin'], 
                marker='o', linewidth=2, label='Grid Feedin', color=self.colors[1])
        
        # Add peak indicators
        max_purchase_hour = hourly_avg.loc[hourly_avg['grid_purchase'].idxmax()]
        max_feedin_hour = hourly_avg.loc[hourly_avg['grid_feedin'].idxmax()]
        
        ax.annotate(f"Peak Purchase: {max_purchase_hour['grid_purchase']:.1f}",
                   xy=(max_purchase_hour['hour_of_day'], max_purchase_hour['grid_purchase']),
                   xytext=(5, 15), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color='black'))
        
        ax.annotate(f"Peak Feedin: {max_feedin_hour['grid_feedin']:.1f}",
                   xy=(max_feedin_hour['hour_of_day'], max_feedin_hour['grid_feedin']),
                   xytext=(5, 15), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color='black'))
        
        # Customize the chart
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Average Energy', fontsize=12)
        ax.set_title('Average Grid Activity by Hour of Day', fontsize=14, fontweight='bold')
        ax.set_xticks(range(0, 24))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add a light gray rectangle for night hours
        ax.axvspan(0, 6, alpha=0.2, color='gray')
        ax.axvspan(18, 24, alpha=0.2, color='gray')
        
        # Add text for day/night
        ax.text(3, ax.get_ylim()[1] * 0.9, 'Night', fontsize=10, ha='center')
        ax.text(12, ax.get_ylim()[1] * 0.9, 'Day', fontsize=10, ha='center')
        ax.text(21, ax.get_ylim()[1] * 0.9, 'Night', fontsize=10, ha='center')
        
        fig.tight_layout()
        self.save_figure(fig, 'hourly_grid_activity.png')
    
    def plot_total_grid_metrics_by_hour(self):
        """Create a bar chart of total grid purchase vs. feedin by hour."""
        logger.info("Generating total grid metrics by hour chart")
        
        # Ensure we have the total metrics columns
        if 'total_grid_purchase' not in self.df.columns or 'total_grid_feedin' not in self.df.columns:
            logger.warning("Total grid metrics columns not found")
            return
        
        # Get unique combinations of date and hour
        hourly_totals = self.df[['date', 'hour_of_day', 'total_grid_purchase', 'total_grid_feedin']].drop_duplicates()
        
        # Group by hour
        hourly_avg = hourly_totals.groupby('hour_of_day').agg({
            'total_grid_purchase': 'mean',
            'total_grid_feedin': 'mean'
        }).reset_index()
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(hourly_avg))
        width = 0.35
        
        ax.bar(x - width/2, hourly_avg['total_grid_purchase'], width, label='Total Grid Purchase', color=self.colors[0])
        ax.bar(x + width/2, hourly_avg['total_grid_feedin'], width, label='Total Grid Feedin', color=self.colors[1])
        
        # Highlight hour with maximum grid feedin
        max_feedin_idx = hourly_avg['total_grid_feedin'].idxmax()
        max_feedin_hour = hourly_avg.iloc[max_feedin_idx]['hour_of_day']
        ax.bar(max_feedin_idx + width/2, hourly_avg.iloc[max_feedin_idx]['total_grid_feedin'], 
               width, label='_nolegend_', color=self.colors[4], alpha=0.7)
        
        ax.annotate(f"Max Feedin Hour: {int(max_feedin_hour)}",
                   xy=(max_feedin_idx + width/2, hourly_avg.iloc[max_feedin_idx]['total_grid_feedin']),
                   xytext=(0, 20), textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color='black'),
                   ha='center', fontweight='bold')
        
        # Customize the chart
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Average Total Energy', fontsize=12)
        ax.set_title('Average Total Grid Metrics by Hour of Day', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(hourly_avg['hour_of_day'])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        self.save_figure(fig, 'total_grid_metrics_by_hour.png')
    
    def plot_grid_metrics_distribution(self):
        """Create histograms of grid purchase and feedin distributions."""
        logger.info("Generating grid metrics distribution chart")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Grid Purchase Distribution (excluding zeros)
        purchase_data = self.df[self.df['grid_purchase'] > 0]['grid_purchase']
        if len(purchase_data) > 0:
            sns.histplot(purchase_data, kde=True, ax=ax1, color=self.colors[0])
            ax1.set_title('Grid Purchase Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Grid Purchase')
            ax1.set_ylabel('Frequency')
        else:
            ax1.text(0.5, 0.5, 'No positive grid purchase data available', 
                    horizontalalignment='center', verticalalignment='center')
        
        # Grid Feedin Distribution (excluding zeros)
        feedin_data = self.df[self.df['grid_feedin'] > 0]['grid_feedin']
        if len(feedin_data) > 0:
            sns.histplot(feedin_data, kde=True, ax=ax2, color=self.colors[1])
            ax2.set_title('Grid Feedin Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Grid Feedin')
            ax2.set_ylabel('Frequency')
        else:
            ax2.text(0.5, 0.5, 'No positive grid feedin data available', 
                    horizontalalignment='center', verticalalignment='center')
        
        fig.suptitle('Distribution of Grid Metrics', fontsize=14, fontweight='bold')
        fig.tight_layout()
        self.save_figure(fig, 'grid_metrics_distribution.png')
    
    def plot_battery_comparison(self):
        """Create a bar chart comparing grid metrics across batteries."""
        logger.info("Generating battery comparison chart")
        
        # Group by serial number (battery ID)
        battery_metrics = self.df.groupby('serial').agg({
            'grid_purchase': 'sum',
            'grid_feedin': 'sum'
        }).reset_index()
        
        # Sort by total activity (purchase + feedin)
        battery_metrics['total_activity'] = battery_metrics['grid_purchase'] + battery_metrics['grid_feedin']
        battery_metrics = battery_metrics.sort_values('total_activity', ascending=False)
        
        # Take top 10 batteries if there are many
        if len(battery_metrics) > 10:
            battery_metrics = battery_metrics.head(10)
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(battery_metrics))
        width = 0.4
        
        ax.bar(x - width/2, battery_metrics['grid_purchase'], width, label='Grid Purchase', color=self.colors[0])
        ax.bar(x + width/2, battery_metrics['grid_feedin'], width, label='Grid Feedin', color=self.colors[1])
        
        # Calculate purchase to feedin ratio
        battery_metrics['purchase_feedin_ratio'] = battery_metrics['grid_purchase'] / battery_metrics['grid_feedin'].replace(0, 1)
        
        # Annotate interesting batteries
        for i, (_, row) in enumerate(battery_metrics.iterrows()):
            if row['grid_feedin'] > row['grid_purchase']:
                ax.annotate('Net Producer', 
                           xy=(i + width/2, row['grid_feedin']),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=8, color='green')
            elif row['grid_purchase'] > 10 * row['grid_feedin'] and row['grid_feedin'] > 0:
                ax.annotate('Heavy Consumer', 
                           xy=(i - width/2, row['grid_purchase']),
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=8, color='red')
        
        # Customize the chart
        ax.set_xlabel('Battery ID (Serial)', fontsize=12)
        ax.set_ylabel('Total Energy', fontsize=12)
        ax.set_title('Grid Metrics by Battery', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(battery_metrics['serial'], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        self.save_figure(fig, 'battery_comparison.png')
    
    def plot_grid_activity_heatmap(self):
        """Create a heatmap of grid activity by hour and battery."""
        logger.info("Generating grid activity heatmap")
        
        # Prepare data for heatmap
        # Sum grid_purchase and grid_feedin to get total grid activity
        self.df['total_grid_activity'] = self.df['grid_purchase'] + self.df['grid_feedin']
        
        # Get top 10 most active batteries
        top_batteries = self.df.groupby('serial')['total_grid_activity'].sum().nlargest(10).index.tolist()
        
        # Filter for these batteries
        df_top = self.df[self.df['serial'].isin(top_batteries)]
        
        # Create pivot table: hours in rows, batteries in columns
        pivot_data = df_top.pivot_table(
            values='total_grid_activity', 
            index='hour_of_day',
            columns='serial',
            aggfunc='mean'
        ).fillna(0)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(14, 10))
        
        sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax, linewidths=.5, 
                   annot=False, fmt='.0f', cbar_kws={'label': 'Average Grid Activity'})
        
        # Customize the chart
        ax.set_title('Grid Activity Heatmap by Hour and Battery', fontsize=14, fontweight='bold')
        ax.set_xlabel('Battery ID (Serial)', fontsize=12)
        ax.set_ylabel('Hour of Day', fontsize=12)
        
        fig.tight_layout()
        self.save_figure(fig, 'grid_activity_heatmap.png')
    
    def plot_time_series(self):
        """Create a time series plot of grid metrics."""
        logger.info("Generating time series chart")
        
        # Ensure we have timestamp data
        if 'timestamp' not in self.df.columns:
            logger.warning("Timestamp column not found")
            return
        
        # Group by timestamp to get average metrics over time
        try:
            # Extract hour from timestamp
            self.df['hour'] = self.df['timestamp'].dt.hour
            
            # Create a combined time index (date and hour)
            self.df['time_index'] = self.df['date'].dt.strftime('%Y-%m-%d') + ' ' + self.df['hour'].astype(str).str.zfill(2) + ':00'
            
            # Sort by this time index
            time_data = self.df.sort_values('time_index')
            
            # Group by time index and calculate average metrics
            time_metrics = time_data.groupby('time_index').agg({
                'grid_purchase': 'mean',
                'grid_feedin': 'mean',
                'timestamp': 'min'  # Keep the timestamp for plotting
            }).reset_index()
            
            # Plot the time series
            fig, ax = plt.subplots(figsize=(15, 7))
            
            ax.plot(time_metrics['timestamp'], time_metrics['grid_purchase'], 
                   label='Grid Purchase', color=self.colors[0], alpha=0.7)
            ax.plot(time_metrics['timestamp'], time_metrics['grid_feedin'], 
                   label='Grid Feedin', color=self.colors[1], alpha=0.7)
            
            # Format x-axis to show date and hour
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45)
            
            # Customize the chart
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Average Energy', fontsize=12)
            ax.set_title('Grid Metrics Time Series', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            self.save_figure(fig, 'grid_metrics_time_series.png')
            
        except Exception as e:
            logger.error(f"Error creating time series chart: {str(e)}")
    
    def plot_feedin_proportion(self):
        """Create a pie chart of battery proportions in total grid feedin."""
        logger.info("Generating feedin proportion chart")
        
        # Group by serial and sum feedin
        battery_feedin = self.df.groupby('serial')['grid_feedin'].sum().sort_values(ascending=False)
        
        # Combine small contributions
        threshold = battery_feedin.sum() * 0.05  # 5% threshold
        small_batteries = battery_feedin[battery_feedin < threshold]
        
        if not small_batteries.empty:
            # Create a new series with small contributions combined
            major_batteries = battery_feedin[battery_feedin >= threshold]
            combined_data = pd.concat([major_batteries, pd.Series({'Others': small_batteries.sum()})])
        else:
            combined_data = battery_feedin
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Only show pie chart if there's actual feedin
        if combined_data.sum() > 0:
            wedges, texts, autotexts = ax.pie(
                combined_data, 
                labels=combined_data.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=sns.color_palette('Set3', len(combined_data)),
                wedgeprops={'edgecolor': 'w', 'linewidth': 1}
            )
            
            # Customize text appearance
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('black')
            
            ax.set_title('Proportion of Grid Feedin by Battery', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No grid feedin data available', 
                   horizontalalignment='center', verticalalignment='center')
        
        fig.tight_layout()
        self.save_figure(fig, 'feedin_proportion.png')
    
    def generate_all_charts(self):
        """Generate all visualization charts."""
        logger.info("Starting visualization generation")
        
        if not self.load_data():
            logger.error("Failed to load data, exiting")
            return False
        
        # Generate all charts
        self.plot_hourly_grid_activity()
        self.plot_total_grid_metrics_by_hour()
        self.plot_grid_metrics_distribution()
        self.plot_battery_comparison()
        self.plot_grid_activity_heatmap()
        self.plot_time_series()
        self.plot_feedin_proportion()
        
        logger.info("All visualizations completed successfully")
        return True


if __name__ == "__main__":
    # Initialize and run the visualizer
    visualizer = BatteryDataVisualizer()
    success = visualizer.generate_all_charts()
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)
