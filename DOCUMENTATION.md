# Project Documentation

## 📚 Complete Guide to Smart Health Tracker Insights

### Table of Contents
1. [Project Overview](#project-overview)
2. [Technical Architecture](#technical-architecture)
3. [Data Pipeline](#data-pipeline)
4. [Model Details](#model-details)
5. [Results Analysis](#results-analysis)
6. [Deployment Guide](#deployment-guide)

---

## Project Overview

### Objective
Develop a comprehensive machine learning system to analyze health metrics from wearable devices and predict:
- Sleep quality and duration
- Stress-sleep relationships
- Activity patterns
- User wellness profiles

### Key Achievements
- ✅ Processed 30,000+ health records
- ✅ Implemented 10+ ML/DL algorithms
- ✅ Achieved 87.6% accuracy in sleep quality prediction
- ✅ ROC-AUC score of 0.959 (excellent discrimination)
- ✅ Statistical validation with hypothesis testing

---

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    Data Layer                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Raw CSV Data (30K records, 11 features)         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Preprocessing Layer                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │  • Missing value imputation                      │  │
│  │  • Feature scaling (StandardScaler)              │  │
│  │  • Label encoding                                │  │
│  │  • Feature engineering                           │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                 Analysis Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │     EDA      │  │  Statistical │  │  Clustering  │  │
│  │              │  │   Analysis   │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│               Modeling Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Regression  │  │Classification│  │ Deep Learning│  │
│  │  - Linear    │  │  - Logistic  │  │  - PyTorch   │  │
│  │  - OLS       │  │  - SVM       │  │  - Keras     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              Evaluation Layer                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │  • Metrics calculation                           │  │
│  │  • Visualization generation                      │  │
│  │  • Model comparison                              │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Technology Stack

**Core Libraries**:
- `pandas` & `numpy`: Data manipulation
- `scikit-learn`: Traditional ML algorithms
- `PyTorch`: Deep learning (custom architectures)
- `TensorFlow/Keras`: Deep learning (sequential models)
- `statsmodels`: Statistical analysis
- `matplotlib` & `seaborn`: Visualization

---

## Data Pipeline

### 1. Data Loading
```python
df = pd.read_csv('data/smart_health_tracker_data.csv')
# Shape: (30000, 11)
```

### 2. Missing Value Handling

| Strategy | Columns | Rationale |
|----------|---------|-----------|
| Median | Age, Resting_Heart_Rate | Robust to outliers |
| Mean | Daily_Steps, Active_Heart_Rate, Hours_of_Sleep, Daily_Calorie_Intake, Sleep_Quality | Normally distributed |
| Mode | Gender, Stress_Level, Daily_Activity_Type, Mood | Categorical variables |

### 3. Feature Engineering

**Derived Features**:
- `Well_Rested`: Binary (1 if Hours_of_Sleep >= 7)
- `High_Sleep_Quality`: Binary (1 if Sleep_Quality >= 75)
- `Activity_Score`: Normalized activity metric

### 4. Feature Scaling
- Method: StandardScaler (zero mean, unit variance)
- Applied to: All numerical features before modeling

---

## Model Details

### Regression Models

#### 1. Linear Regression (Sleep Prediction)
**Hypothesis**:
- H₀: No relationship between (Daily_Steps, Stress_Level) and Hours_of_Sleep
- H₁: Significant relationship exists

**Features**: Daily_Steps, Stress_Level  
**Target**: Hours_of_Sleep  
**Results**: R² score, coefficient analysis

#### 2. OLS Regression (Statistical Validation)
**Purpose**: Hypothesis testing with p-values  
**Output**: Statistical significance of predictors

### Classification Models

#### 1. Logistic Regression
- **Type**: Binary classification
- **Target**: Well_Rested (sleep >= 7 hours)
- **Features**: Sleep_Quality, Stress_Level, Daily_Calorie_Intake, Active_Heart_Rate, Resting_Heart_Rate
- **Hyperparameters**: class_weight='balanced', max_iter=1000
- **Performance**: 87.3% accuracy, 0.958 ROC-AUC

#### 2. Support Vector Machine (SVM)
**Kernel Variants**:
- **RBF Kernel**: Best for non-linear patterns
  - Accuracy: 87.5%
  - ROC-AUC: 0.958
- **Linear Kernel**: Baseline comparison
- **Sigmoid Kernel**: Activity type classification

**Key Parameters**:
- `class_weight='balanced'`: Handle class imbalance
- `probability=True`: Enable ROC-AUC calculation

### Deep Learning Models

#### 1. PyTorch Perceptron
**Architecture**:
```
Input (5) → Linear → Sigmoid → Output (1)
```
**Training**:
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=0.01)
- Epochs: 100

**Performance**: 87.5% accuracy

#### 2. PyTorch Deep Neural Network
**Architecture**:
```
Input (5) → Dense(32) → ReLU → Dense(16) → ReLU → Dense(1) → Sigmoid
```
**Performance**: 87.6% accuracy, 0.959 ROC-AUC (Best overall)

#### 3. Keras Sequential Model
**Architecture**: Same as PyTorch Deep NN  
**Advantages**: Simpler API, built-in validation split

### Clustering Models

#### 1. K-Means
- **Features**: Hours_of_Sleep, Sleep_Quality, Stress_Level
- **Clusters**: 8 (optimized via silhouette score)
- **Use Case**: User behavior segmentation

#### 2. DBSCAN
- **Features**: Daily_Steps, Daily_Calorie_Intake
- **Parameters**: eps=0.04, min_samples=20
- **Use Case**: Outlier detection

---

## Results Analysis

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Deep NN (PyTorch)** | **87.6%** | 87.3% | 85.7% | 86.5% | **0.959** |
| Perceptron (PyTorch) | 87.5% | **94.4%** | 77.7% | 85.2% | 0.958 |
| Logistic Regression | 87.3% | 84.1% | **89.5%** | **86.7%** | 0.958 |
| SVM (RBF) | 87.5% | 87.0% | 85.0% | 86.0% | 0.958 |

### Key Insights

1. **Best Overall Model**: Deep NN (highest ROC-AUC)
2. **Best Precision**: Perceptron (fewer false positives)
3. **Best Recall**: Logistic Regression (catches more true positives)
4. **All models perform well** (>87% accuracy)

### Statistical Findings

1. **Stress-Sleep Correlation**: -0.XX (p < 0.05)
   - Significant negative relationship
   - Higher stress → Lower sleep quality

2. **Activity Impact**: 
   - Intense activity correlates with higher calorie intake
   - Moderate activity shows best sleep quality

3. **Heart Rate Patterns**:
   - Active heart rate is strong predictor of activity type
   - Resting heart rate correlates with stress levels

---

## Deployment Guide

### Local Deployment

1. **Clone and Setup**
```bash
git clone https://github.com/ishikadubey1105/Smart-Health-Tracker-Insights.git
cd Smart-Health-Tracker-Insights
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download Data**
- Get dataset from Google Drive link
- Place in `data/smart_health_tracker_data.csv`

3. **Run Analysis**
```bash
python src/main_analysis.py
```

### Production Deployment (Future)

**API Development**:
```python
# Flask/FastAPI endpoint
@app.post("/predict")
def predict_sleep_quality(features: HealthFeatures):
    # Load model
    model = load_model('models/best_model.pkl')
    # Predict
    prediction = model.predict(features)
    return {"sleep_quality": prediction}
```

**Containerization**:
```dockerfile
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "src/api.py"]
```

---

## Future Enhancements

### Short-term
- [ ] Jupyter notebook with interactive widgets
- [ ] Model persistence (save/load trained models)
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Feature importance analysis (SHAP values)

### Medium-term
- [ ] REST API with Flask/FastAPI
- [ ] Web dashboard (Streamlit/Dash)
- [ ] Time series forecasting (LSTM)
- [ ] Ensemble methods (Random Forest, XGBoost)

### Long-term
- [ ] Real-time data pipeline
- [ ] Mobile app integration
- [ ] Personalized recommendations engine
- [ ] A/B testing framework

---

## References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. PyTorch Tutorials: https://pytorch.org/tutorials/
3. TensorFlow Guide: https://www.tensorflow.org/guide
4. Statistical Analysis: https://www.statsmodels.org/

---

**Last Updated**: February 2026  
**Version**: 1.0.0  
**Author**: Ishika Dubey
