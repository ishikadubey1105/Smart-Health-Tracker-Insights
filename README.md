# Smart Health Tracker Insights

Predicting sleep quality from wearable fitness data, and comparing a deep network against
simpler baselines to see whether the extra complexity is actually earning its place.

**Headline result: it barely is.** A deep neural network reaches 87.6% accuracy; logistic
regression reaches 87.3%. That 0.3-point gap is the interesting finding, and it is discussed
below rather than buried.

Built for the Minor in Artificial Intelligence (Batch 4), Mini Project Modules A & B.

---

## The data

11 features across activity, cardiac and sleep measurements. [Dataset link](https://drive.google.com/file/d/15RtFohMLHMCzPaMFv7nfA2xu8R5mbgff/view?usp=sharing).

| Feature | Type |
| --- | --- |
| Age | numeric |
| Gender | categorical |
| Daily steps | numeric |
| Resting heart rate | numeric (bpm) |
| Active heart rate | numeric (bpm) |
| Hours of sleep | numeric |
| Daily calorie intake | numeric |
| Stress level | ordinal (0–10) |
| Sleep quality | numeric (0–100) |
| Daily activity type | categorical (sedentary / moderate / intense) |
| Mood | categorical (sad / neutral / happy) |

**Preprocessing:** median imputation for numeric and mode for categorical, `StandardScaler`
on numeric features, label encoding for categoricals, stratified 80/20 split.

## Results — sleep quality (binary classification)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Deep Neural Network | **87.6%** | 87.3% | 85.7% | 86.5% | **0.959** |
| Perceptron | 87.5% | **94.4%** | 77.7% | 85.2% | 0.958 |
| Logistic Regression | 87.3% | 84.1% | **89.5%** | **86.7%** | 0.958 |

### What this table actually says

- **The deep network is not meaningfully better.** It wins accuracy by 0.3 points and ROC-AUC
  by 0.001 over logistic regression. On a dataset this size that difference is well inside
  noise, and logistic regression actually has the better F1. If this were a system anyone had
  to maintain, logistic regression would be the correct choice — it trains in a fraction of
  the time and you can read its coefficients.
- **The models differ in *how* they are right, not how often.** The Perceptron has the highest
  precision (94.4%) and the lowest recall (77.7%) — it is conservative, flagging fewer cases
  but being right more often when it does. Logistic regression is the opposite. Which one you
  want depends entirely on whether a missed case or a false alarm costs more.
- All three sit around 87%, which suggests the ceiling here is the data, not the model.

### Other findings

- **Stress and sleep quality are strongly negatively correlated** (p < 0.05) — higher stress,
  worse sleep. This is the clearest signal in the dataset.
- **Active heart rate is a strong predictor of activity type.**
- **Class balancing** (`class_weight='balanced'`) improved results measurably.
- **Clustering:** K-Means separated the users into 8 behaviour profiles by sleep pattern;
  DBSCAN flagged outliers in the step-to-calorie relationship.

## Limitations

- The dataset is modest, which is why I treat sub-1-point differences as noise rather than
  ranking the models by them
- Sleep quality is a derived score, not a measured sleep-lab outcome
- No hyperparameter search — the deep network's architecture was chosen, not tuned, so its
  ceiling here is not established
- Correlational throughout; nothing here supports a causal claim about stress and sleep

## Running it

```bash
git clone https://github.com/ishikadubey1105/Smart-Health-Tracker-Insights
cd Smart-Health-Tracker-Insights
pip install -r requirements.txt

jupyter notebook notebooks/
```

Download the dataset from the link above into `data/` first.

## Structure

```
src/           model and preprocessing code
notebooks/     EDA, modelling and clustering
data/          dataset (not committed)
assets/        generated charts
DOCUMENTATION.md
QUICKSTART.md
```

---

**Stack:** Python, scikit-learn, PyTorch, TensorFlow, pandas, statsmodels, Matplotlib, Seaborn

*Built by [Ishika Dubey](https://github.com/ishikadubey1105) — B.Tech CSE (AI & ML), SIT Nagpur*
