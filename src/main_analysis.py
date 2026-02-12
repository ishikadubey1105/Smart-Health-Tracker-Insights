"""
Main Analysis Script
Runs the complete Smart Health Tracker analysis pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from preprocessing import load_and_preprocess_data
from eda import perform_eda
from classification_models import train_sleep_quality_classifiers
from deep_learning_models import train_deep_learning_models

def main():
    """Run complete analysis pipeline"""
    
    print("\n" + "="*70)
    print("SMART HEALTH TRACKER INSIGHTS - COMPLETE ANALYSIS")
    print("="*70)
    
    # Step 1: Load and preprocess data
    print("\n[STEP 1/4] Loading and preprocessing data...")
    data_path = "../data/smart_health_tracker_data.csv"
    df, preprocessor = load_and_preprocess_data(data_path, verbose=True)
    
    # Step 2: Exploratory Data Analysis
    print("\n[STEP 2/4] Performing exploratory data analysis...")
    perform_eda(df, save_dir='../assets')
    
    # Step 3: Train classification models
    print("\n[STEP 3/4] Training classification models...")
    classifier = train_sleep_quality_classifiers(df, save_dir='../results')
    
    # Step 4: Train deep learning models
    print("\n[STEP 4/4] Training deep learning models...")
    dl_trainer = train_deep_learning_models(df, save_dir='../results')
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📁 Results saved to:")
    print("   - Visualizations: ../assets/")
    print("   - Model results: ../results/")
    print("\n✅ All tasks completed successfully!")
    print("\nNext steps:")
    print("   1. Review visualizations in the assets/ folder")
    print("   2. Check model performance in results/")
    print("   3. Explore the Jupyter notebook for interactive analysis")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
