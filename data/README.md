# Data Directory

## Dataset Information

This directory contains the Smart Health Tracker dataset used for analysis.

### Download Instructions

1. Download the dataset from: [Google Drive Link](https://drive.google.com/file/d/15RtFohMLHMCzPaMFv7nfA2xu8R5mbgff/view?usp=sharing)
2. Save the file as `smart_health_tracker_data.csv` in this directory

### Dataset Structure

**File**: `smart_health_tracker_data.csv`  
**Size**: ~30,000 rows × 11 columns  
**Format**: CSV (Comma-Separated Values)

### Features

| Column Name | Type | Description | Range/Values |
|------------|------|-------------|--------------|
| Age | Numeric | User age in years | 18-80 |
| Gender | Categorical | User gender | Male, Female |
| Daily_Steps | Numeric | Total steps per day | 0-20,000 |
| Resting_Heart_Rate | Numeric | Heart rate at rest (bpm) | 40-100 |
| Active_Heart_Rate | Numeric | Heart rate during activity (bpm) | 80-180 |
| Hours_of_Sleep | Numeric | Total sleep duration (hours) | 3-10 |
| Daily_Calorie_Intake | Numeric | Calories consumed per day | 1200-4000 |
| Stress_Level | Ordinal | Stress rating | 0-10 |
| Sleep_Quality | Numeric | Sleep quality score | 0-100 |
| Daily_Activity_Type | Categorical | Activity intensity | sedentary, moderate, intense |
| Mood | Categorical | User mood | sad, neutral, happy |

### Data Quality

- **Missing Values**: Present in multiple columns (~5-15% per column)
- **Handling**: Imputed using median (numeric) and mode (categorical)
- **Outliers**: Present but retained as they represent valid health variations

### Usage

```python
import pandas as pd

# Load the data
df = pd.read_csv('data/smart_health_tracker_data.csv')

# Or use the preprocessing module
from src.preprocessing import load_and_preprocess_data
df, preprocessor = load_and_preprocess_data('data/smart_health_tracker_data.csv')
```

### Citation

If you use this dataset, please cite:
```
Smart Health Tracker Dataset (2024)
Wearable Fitness Device Data Collection
```

### License

Dataset is used for educational and research purposes.
