"""
Exploratory Data Analysis Module
Comprehensive visualizations and statistical analysis of health tracker data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class HealthDataEDA:
    """
    Exploratory Data Analysis for Health Tracker Data
    """
    
    def __init__(self, df, save_dir='../assets'):
        """
        Initialize EDA with dataframe
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            save_dir (str): Directory to save plots
        """
        self.df = df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_distributions(self, save=True):
        """Plot distributions of all features"""
        print("\n📊 Creating distribution plots...")
        
        fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(16, 24))
        axes = axes.flatten()
        
        # Plot 1 - Age
        axes[0].hist(self.df['Age'], bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0].set_title('Age Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Age')
        axes[0].set_ylabel('Frequency')
        
        # Plot 2 - Gender
        self.df['Gender'].value_counts().plot(kind='bar', ax=axes[1], color='teal', alpha=0.7)
        axes[1].set_title('Gender Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Gender')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=0)
        
        # Plot 3 - Daily Steps
        axes[2].hist(self.df['Daily_Steps'], bins=20, color='dodgerblue', edgecolor='black', alpha=0.7)
        axes[2].set_title('Daily Steps Distribution', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Frequency')
        
        # Plot 4 - Resting Heart Rate
        axes[3].boxplot(self.df['Resting_Heart_Rate'].dropna(), patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[3].set_title('Resting Heart Rate', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('BPM')
        
        # Plot 5 - Active Heart Rate
        axes[4].boxplot(self.df['Active_Heart_Rate'].dropna(), patch_artist=True,
                       boxprops=dict(facecolor='salmon', alpha=0.7))
        axes[4].set_title('Active Heart Rate', fontsize=12, fontweight='bold')
        axes[4].set_ylabel('BPM')
        
        # Plot 6 - Hours of Sleep
        axes[5].hist(self.df['Hours_of_Sleep'], bins=10, color='mediumseagreen', 
                    edgecolor='black', alpha=0.7)
        axes[5].set_title('Hours of Sleep Distribution', fontsize=12, fontweight='bold')
        axes[5].set_xlabel('Hours')
        axes[5].set_ylabel('Frequency')
        
        # Plot 7 - Calorie Intake
        axes[6].hist(self.df['Daily_Calorie_Intake'], bins=20, color='orange', 
                    edgecolor='black', alpha=0.7)
        axes[6].set_title('Daily Calorie Intake', fontsize=12, fontweight='bold')
        axes[6].set_xlabel('Calories')
        axes[6].set_ylabel('Frequency')
        
        # Plot 8 - Stress Level
        self.df['Stress_Level'].value_counts().sort_index().plot(
            kind='bar', ax=axes[7], color='orchid', alpha=0.7)
        axes[7].set_title('Stress Level Distribution (0-10)', fontsize=12, fontweight='bold')
        axes[7].set_xlabel('Stress Level')
        axes[7].set_ylabel('Count')
        
        # Plot 9 - Sleep Quality
        axes[8].hist(self.df['Sleep_Quality'], bins=10, color='slateblue', 
                    edgecolor='black', alpha=0.7)
        axes[8].set_title('Sleep Quality Score', fontsize=12, fontweight='bold')
        axes[8].set_xlabel('Quality Score')
        axes[8].set_ylabel('Frequency')
        
        # Plot 10 - Activity Type
        self.df['Daily_Activity_Type'].value_counts().plot(
            kind='bar', ax=axes[9], color='goldenrod', alpha=0.7)
        axes[9].set_title('Activity Type Distribution', fontsize=12, fontweight='bold')
        axes[9].set_xlabel('Activity Type')
        axes[9].set_ylabel('Count')
        axes[9].tick_params(axis='x', rotation=45)
        
        # Plot 11 - Mood
        self.df['Mood'].value_counts().plot(kind='bar', ax=axes[10], color='lightgreen', alpha=0.7)
        axes[10].set_title('Mood Distribution', fontsize=12, fontweight='bold')
        axes[10].set_xlabel('Mood')
        axes[10].set_ylabel('Count')
        axes[10].tick_params(axis='x', rotation=0)
        
        # Remove last unused subplot
        fig.delaxes(axes[11])
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'eda_overview.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved distribution plots to {save_path}")
        
        plt.show()
        
    def plot_correlation_heatmap(self, save=True):
        """Plot correlation heatmap of numerical features"""
        print("\n📊 Creating correlation heatmap...")
        
        plt.figure(figsize=(12, 10))
        corr_matrix = self.df.corr(numeric_only=True)
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                   linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Health Features', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'correlation_heatmap.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved correlation heatmap to {save_path}")
        
        plt.show()
        
    def plot_stress_sleep_relationship(self, save=True):
        """Analyze stress vs sleep quality relationship"""
        print("\n📊 Analyzing stress-sleep relationship...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot with regression line
        sns.regplot(x='Stress_Level', y='Sleep_Quality', data=self.df, 
                   scatter_kws={'alpha':0.3}, ax=axes[0], color='purple')
        axes[0].set_title('Stress Level vs Sleep Quality', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Stress Level')
        axes[0].set_ylabel('Sleep Quality')
        axes[0].grid(True, alpha=0.3)
        
        # Colored by mood
        sns.scatterplot(x='Stress_Level', y='Sleep_Quality', hue='Mood', 
                       data=self.df, ax=axes[1], alpha=0.6)
        axes[1].set_title('Stress vs Sleep Quality (by Mood)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Stress Level')
        axes[1].set_ylabel('Sleep Quality')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'stress_sleep_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved stress-sleep analysis to {save_path}")
        
        plt.show()
        
    def plot_activity_analysis(self, save=True):
        """Analyze activity patterns"""
        print("\n📊 Analyzing activity patterns...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Calorie intake by activity type
        sns.boxplot(x='Daily_Activity_Type', y='Daily_Calorie_Intake', 
                   data=self.df, ax=axes[0], palette='Set2')
        axes[0].set_title('Calorie Intake by Activity Type', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Activity Type')
        axes[0].set_ylabel('Daily Calorie Intake')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Steps vs Calorie Intake
        sns.scatterplot(x='Daily_Steps', y='Daily_Calorie_Intake', 
                       data=self.df, ax=axes[1], alpha=0.5, color='steelblue')
        axes[1].set_title('Daily Steps vs Calorie Intake', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Daily Steps')
        axes[1].set_ylabel('Calorie Intake')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / 'activity_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved activity analysis to {save_path}")
        
        plt.show()
        
    def plot_pairplot(self, save=True):
        """Create pairplot of key features"""
        print("\n📊 Creating pairplot (this may take a moment)...")
        
        key_features = ['Sleep_Quality', 'Stress_Level', 'Hours_of_Sleep', 'Mood']
        pairplot = sns.pairplot(self.df[key_features], hue='Mood', 
                               plot_kws={'alpha': 0.6}, diag_kind='kde')
        pairplot.fig.suptitle('Pairwise Relationships of Key Health Metrics', 
                             y=1.02, fontsize=16, fontweight='bold')
        
        if save:
            save_path = self.save_dir / 'pairplot.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved pairplot to {save_path}")
        
        plt.show()
        
    def generate_all_plots(self):
        """Generate all EDA visualizations"""
        print("\n" + "="*60)
        print("GENERATING ALL EDA VISUALIZATIONS")
        print("="*60)
        
        self.plot_distributions()
        self.plot_correlation_heatmap()
        self.plot_stress_sleep_relationship()
        self.plot_activity_analysis()
        self.plot_pairplot()
        
        print("\n✅ All visualizations generated successfully!")
        print(f"📁 Saved to: {self.save_dir.absolute()}")


def perform_eda(df, save_dir='../assets'):
    """
    Convenience function to perform complete EDA
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe
        save_dir (str): Directory to save plots
    """
    eda = HealthDataEDA(df, save_dir)
    eda.generate_all_plots()
    return eda


if __name__ == "__main__":
    # Example usage
    from preprocessing import load_and_preprocess_data
    
    df, _ = load_and_preprocess_data("../data/smart_health_tracker_data.csv")
    perform_eda(df)
