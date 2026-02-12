# Assets Directory

This directory contains all visualizations and images generated from the analysis.

## Contents

After running the analysis, this folder will contain:

### Exploratory Data Analysis
- `eda_overview.png` - Comprehensive distribution plots of all features
- `correlation_heatmap.png` - Correlation matrix of numerical features
- `stress_sleep_analysis.png` - Stress vs sleep quality relationships
- `activity_analysis.png` - Activity patterns and calorie intake analysis
- `pairplot.png` - Pairwise relationships of key health metrics

### Model Performance
- `confusion_matrix_*.png` - Confusion matrices for each model
- `roc_curve_*.png` - ROC curves for each classifier
- `model_comparison.png` - Bar chart comparing all models

### Architecture Diagrams
- `architecture_diagram.png` - System architecture overview (if created)

## Usage in README

These images are referenced in the main README.md file to showcase project results.

## Generating Assets

Run the complete analysis to generate all visualizations:

```bash
python src/main_analysis.py
```

Or generate specific visualizations:

```bash
# EDA only
python src/eda.py

# Classification models
python src/classification_models.py

# Deep learning models
python src/deep_learning_models.py
```

---

**Note**: This folder is tracked in git but the actual image files are ignored (.gitignore). 
You'll need to run the analysis to generate them locally.
