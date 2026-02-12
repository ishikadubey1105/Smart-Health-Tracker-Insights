# 🚀 Quick Start Guide

Get up and running with Smart Health Tracker Insights in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Git
- 2GB free disk space

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/ishikadubey1105/Smart-Health-Tracker-Insights.git
cd Smart-Health-Tracker-Insights
```

### 2. Set Up Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas, numpy (data processing)
- scikit-learn (ML algorithms)
- torch, tensorflow (deep learning)
- matplotlib, seaborn (visualization)
- statsmodels (statistical analysis)

### 4. Download the Dataset

1. Visit: https://drive.google.com/file/d/15RtFohMLHMCzPaMFv7nfA2xu8R5mbgff/view?usp=sharing
2. Download `smart_health_tracker_data (1).csv`
3. Rename it to `smart_health_tracker_data.csv`
4. Place it in the `data/` folder

Your directory should look like:
```
Smart-Health-Tracker-Insights/
├── data/
│   └── smart_health_tracker_data.csv  ← Your downloaded file
├── src/
├── ...
```

### 5. Run the Analysis

**Option A: Complete Analysis (Recommended for first run)**
```bash
python src/main_analysis.py
```

This will:
- ✅ Load and preprocess data
- ✅ Generate all visualizations
- ✅ Train all ML models
- ✅ Train all DL models
- ✅ Save results to `results/` and `assets/`

**Estimated time**: 5-10 minutes

**Option B: Individual Modules**

```bash
# Just EDA
python src/eda.py

# Just classification models
python src/classification_models.py

# Just deep learning
python src/deep_learning_models.py
```

### 6. View Results

After running the analysis:

**Visualizations**: Check `assets/` folder
- `eda_overview.png` - Data distributions
- `correlation_heatmap.png` - Feature correlations
- `confusion_matrix_*.png` - Model performance
- `roc_curve_*.png` - ROC curves

**Model Metrics**: Check `results/` folder
- `model_comparison.csv` - Performance comparison
- Various PNG files with detailed metrics

## Jupyter Notebook (Interactive)

For interactive exploration:

```bash
jupyter notebook notebooks/original_analysis.py
```

Or create a new notebook and import modules:

```python
from src.preprocessing import load_and_preprocess_data
from src.eda import perform_eda
from src.classification_models import train_sleep_quality_classifiers

# Load data
df, preprocessor = load_and_preprocess_data('data/smart_health_tracker_data.csv')

# Explore
perform_eda(df)

# Train models
classifier = train_sleep_quality_classifiers(df)
```

## Troubleshooting

### Issue: Module not found

**Solution**: Make sure you're in the project root and virtual environment is activated

```bash
cd Smart-Health-Tracker-Insights
# Activate venv
pip install -r requirements.txt
```

### Issue: File not found error

**Solution**: Ensure dataset is in the correct location

```bash
# Check if file exists
ls data/smart_health_tracker_data.csv  # macOS/Linux
dir data\smart_health_tracker_data.csv  # Windows
```

### Issue: Memory error

**Solution**: Close other applications or reduce dataset size for testing

```python
# In preprocessing.py, add:
df = df.sample(n=10000, random_state=42)  # Use smaller sample
```

### Issue: Slow training

**Solution**: Reduce epochs for deep learning models

```python
# In deep_learning_models.py, change:
epochs=50  # Instead of 100
```

## Next Steps

1. ✅ **Explore the code**: Check out `src/` modules
2. ✅ **Read documentation**: See `DOCUMENTATION.md` for details
3. ✅ **Customize models**: Modify hyperparameters
4. ✅ **Add features**: Contribute via pull requests!

## Quick Commands Reference

```bash
# Activate environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Run complete analysis
python src/main_analysis.py

# Run specific module
python src/eda.py
python src/classification_models.py
python src/deep_learning_models.py

# Start Jupyter
jupyter notebook

# Deactivate environment
deactivate
```

## Getting Help

- 📖 Read the [full documentation](DOCUMENTATION.md)
- 🐛 Report issues on [GitHub Issues](https://github.com/ishikadubey1105/Smart-Health-Tracker-Insights/issues)
- 💬 Check [Contributing Guide](CONTRIBUTING.md)

---

**Ready to go?** Run `python src/main_analysis.py` and watch the magic happen! ✨
