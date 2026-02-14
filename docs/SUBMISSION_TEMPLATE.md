# ML Assignment 2 - Submission PDF Template
## Copy this into Word/Google Docs, fill placeholders, export as PDF

---

# Machine Learning Assignment 2 - Classification Models

**Student Name:** [Your Full Name]
**Student ID:** [Your Student ID]
**Program:** M.Tech (AIML)
**Course:** Machine Learning
**Submission Date:** [Date]

---

## 1. GitHub Repository Link

**Repository URL:**
```
https://github.com/[yourusername]/[your-repo-name]
```

**Repository Contents:**
- app.py — Streamlit web application
- requirements.txt — Python dependencies
- README.md — Complete documentation
- .gitignore
- model/model_training.ipynb — Jupyter notebook
- input/adult_income.csv — UCI Adult Income dataset
- output/ — Generated results and charts
- utils/ml_utils.py — Shared ML utilities
- docs/ — Guides and templates

---

## 2. Live Streamlit App Link

**Application URL:**
```
https://[your-app-name].streamlit.app
```

**Implemented Features:**
- CSV file upload
- Model selection dropdown (6 models)
- All 6 evaluation metrics displayed
- Confusion matrix visualization
- Classification report
- Model comparison table

---

## 3. BITS Virtual Lab Screenshot

[INSERT SCREENSHOT HERE]

Screenshot shows:
- BITS Virtual Lab interface
- model_training.ipynb running
- Results table with all 6 models and all 6 metrics

---

## 4. README Content

---

# Machine Learning Assignment 2 - Classification Models

## Problem Statement

This project implements a comprehensive machine learning classification pipeline featuring six different classification algorithms applied to the UCI Adult Income dataset. The goal is to predict whether an individual's annual income exceeds $50K based on census attributes, and to compare the performance of traditional ML models (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes) against ensemble methods (Random Forest and XGBoost) using six evaluation metrics.

---

## Dataset Description

- **Dataset Name:** UCI Adult Income Dataset (Census Income)
- **Source:** UCI Machine Learning Repository — https://archive.ics.uci.edu/ml/datasets/adult
- **Type:** Binary Classification
- **Number of Instances:** 30,162 — Requirement: ≥ 500 ✓
- **Number of Features:** 14 — Requirement: ≥ 12 ✓
- **Target Variable:** income (<=50K or >50K)
- **Class Distribution:**
  - Class 0 (<=50K): 22,654 samples (75.1%)
  - Class 1 (>50K): 7,508 samples (24.9%)

**Numerical Features (6):** age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week

**Categorical Features (8):** workclass, education, marital_status, occupation, relationship, race, sex, native_country

**Preprocessing:**
1. Label Encoding on all 8 categorical features
2. Target encoded: <=50K → 0, >50K → 1
3. StandardScaler normalization
4. 80-20 stratified train-test split

---

## Models Used

### Comparison Table - Model Performance Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| **Logistic Regression** | 0.8175 | 0.8501 | 0.8060 | 0.8175 | 0.8018 | 0.4613 |
| **Decision Tree** | 0.8508 | 0.8855 | 0.8446 | 0.8508 | 0.8451 | 0.5789 |
| **K-Nearest Neighbors** | 0.8190 | 0.8498 | 0.8133 | 0.8190 | 0.8154 | 0.4993 |
| **Naive Bayes** | 0.7978 | 0.8498 | 0.7830 | 0.7978 | 0.7697 | 0.3798 |
| **Random Forest** | 0.8589 | 0.9136 | 0.8534 | 0.8589 | 0.8526 | 0.6003 |
| **XGBoost** | 0.8671 | 0.9243 | 0.8624 | 0.8671 | 0.8624 | 0.6269 |

---

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Logistic Regression achieves 81.75% accuracy with AUC 0.8501, serving as a solid linear baseline. It performs well because features like education_num, age, and hours_per_week have approximately linear relationships with income. The model converges reliably with lbfgs solver and is the most interpretable among the six, though it falls short on non-linear patterns compared to tree-based models. |
| **Decision Tree** | Decision Tree achieves 85.08% accuracy with AUC 0.8855 and MCC 0.5789. With max_depth=10, it successfully captures non-linear interactions between occupation, marital_status, and education without extreme overfitting. It outperforms linear models significantly while remaining interpretable through decision rules. However, it is surpassed by ensemble methods which reduce its variance. |
| **K-Nearest Neighbors** | KNN with k=5 achieves 81.90% accuracy with AUC 0.8498, slightly edging out Logistic Regression. Feature scaling via StandardScaler is essential for KNN's distance calculations and was correctly applied. On 30,162 instances, it is computationally heavier at prediction time but benefits from the demographic clustering present in the Adult Income dataset. |
| **Naive Bayes** | Gaussian Naive Bayes achieves the lowest accuracy at 79.78% with MCC 0.3798. The feature independence assumption does not hold well here — features like education, occupation, and marital_status are correlated — limiting its performance. Despite this, it achieves a competitive AUC of 0.8498, indicating reasonable probability calibration, and is the fastest model to train. |
| **Random Forest** | Random Forest achieves 85.89% accuracy with AUC 0.9136 and MCC 0.6003, ranking second overall. The ensemble of 100 trees reduces overfitting through bagging and random feature subsets. Its AUC of 0.9136 reflects excellent discriminative ability and it consistently outperforms all traditional models across every metric. |
| **XGBoost** | XGBoost is the best performing model with 86.71% accuracy, highest AUC (0.9243), F1 (0.8624), and MCC (0.6269). Its gradient boosting framework iteratively corrects errors from previous trees, capturing complex non-linear patterns. With learning_rate=0.1, max_depth=6, and 100 estimators, it demonstrates the clear advantage of advanced ensemble techniques over traditional classifiers. |

**Overall Insights:**
- Best performing model: **XGBoost** — 86.71% accuracy, AUC 0.9243
- Ensemble methods (Random Forest and XGBoost) outperform all traditional algorithms across every metric
- Naive Bayes is weakest (79.78%) due to its feature independence assumption not holding for this correlated dataset
- All models achieved AUC > 0.84, indicating good discriminative ability despite class imbalance

---

**Author:** [Your Name] | M.Tech (AIML), BITS Pilani WILP | [Your Email]

---

**END OF SUBMISSION**
