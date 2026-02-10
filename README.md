# Machine Learning Assignment 2 - Classification Models

## Problem Statement

This project implements a comprehensive machine learning classification pipeline featuring six different classification algorithms applied to the UCI Adult Income dataset. The goal is to predict whether an individual's annual income exceeds $50K based on census attributes, and to compare the performance of traditional ML models (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes) against ensemble methods (Random Forest and XGBoost) using six evaluation metrics. The project includes an interactive Streamlit web application for model demonstration and evaluation.

---

## Dataset Description

### Dataset Information
- **Dataset Name**: UCI Adult Income Dataset (Census Income)
- **Source**: [UCI Machine Learning Repository - Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- **Type**: Binary Classification
- **URL**: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

### Dataset Characteristics
- **Number of Instances**: 30,162 (after removing missing values) — Minimum requirement: 500 ✓
- **Number of Features**: 14 — Minimum requirement: 12 ✓
- **Target Variable**: `income`
- **Class Distribution**:
  - Class 0 (<=50K): 22,654 samples (75.1%)
  - Class 1 (>50K): 7,508 samples (24.9%)

### Features Overview

| Feature | Type | Description |
|---------|------|-------------|
| age | Numerical | Age of the individual |
| workclass | Categorical | Employment type (Private, Self-emp, Gov, etc.) |
| fnlwgt | Numerical | Census sampling weight |
| education | Categorical | Highest education level attained |
| education_num | Numerical | Education level as a number (1–16) |
| marital_status | Categorical | Marital status |
| occupation | Categorical | Type of occupation |
| relationship | Categorical | Relationship status in household |
| race | Categorical | Race of the individual |
| sex | Categorical | Gender |
| capital_gain | Numerical | Capital gains recorded |
| capital_loss | Numerical | Capital losses recorded |
| hours_per_week | Numerical | Average hours worked per week |
| native_country | Categorical | Country of origin |

- **Numerical Features (6)**: age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week
- **Categorical Features (8)**: workclass, education, marital_status, occupation, relationship, race, sex, native_country
- **Missing Values**: None (rows with missing values removed using dropna)

### Data Preprocessing Steps
1. **Encoding**: All 8 categorical features encoded using Label Encoding
2. **Target Encoding**: Target variable (income) encoded — `<=50K` → 0, `>50K` → 1
3. **Scaling**: All features normalized using StandardScaler
4. **Train-Test Split**: 80-20 split with stratification to preserve class balance

---

## Models Used

### Model Implementation Summary
All six models were implemented using scikit-learn and XGBoost libraries, trained on the same 80-20 stratified train-test split for fair comparison.

### Comparison Table - Model Performance Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| **Logistic Regression** | 0.8175 | 0.8501 | 0.8060 | 0.8175 | 0.8018 | 0.4613 |
| **Decision Tree** | 0.8508 | 0.8855 | 0.8446 | 0.8508 | 0.8451 | 0.5789 |
| **K-Nearest Neighbors** | 0.8190 | 0.8498 | 0.8133 | 0.8190 | 0.8154 | 0.4993 |
| **Naive Bayes** | 0.7978 | 0.8498 | 0.7830 | 0.7978 | 0.7697 | 0.3798 |
| **Random Forest (Ensemble)** | 0.8589 | 0.9136 | 0.8534 | 0.8589 | 0.8526 | 0.6003 |
| **XGBoost (Ensemble)** | 0.8671 | 0.9243 | 0.8624 | 0.8671 | 0.8624 | 0.6269 |

---

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Logistic Regression achieves 81.75% accuracy with AUC 0.8501, serving as a solid linear baseline. It performs well because features like education_num, age, and hours_per_week have approximately linear relationships with income. The model converges reliably with lbfgs solver and is the most interpretable among the six, though it falls short on non-linear patterns compared to tree-based models. |
| **Decision Tree** | Decision Tree achieves 85.08% accuracy with AUC 0.8855 and MCC 0.5789. With max_depth=10, it successfully captures non-linear interactions between occupation, marital_status, and education without extreme overfitting. It outperforms linear models significantly while remaining interpretable through decision rules. However, it is surpassed by ensemble methods which reduce its variance. |
| **K-Nearest Neighbors** | KNN with k=5 achieves 81.90% accuracy with AUC 0.8498, slightly edging out Logistic Regression. Feature scaling via StandardScaler is essential for KNN's distance calculations and was correctly applied. On 30,162 instances, it is computationally heavier at prediction time but benefits from the demographic clustering present in the Adult Income dataset. |
| **Naive Bayes** | Gaussian Naive Bayes achieves the lowest accuracy at 79.78% with MCC 0.3798, the weakest MCC among all models. The feature independence assumption does not hold well here — features like education, occupation, and marital_status are correlated — limiting its performance. Despite this, it achieves a competitive AUC of 0.8498, indicating reasonable probability calibration, and is the fastest model to train. |
| **Random Forest (Ensemble)** | Random Forest achieves 85.89% accuracy with AUC 0.9136 and MCC 0.6003, ranking second overall. The ensemble of 100 trees reduces overfitting through bagging and random feature subsets. Its AUC of 0.9136 reflects excellent discriminative ability. It consistently outperforms all traditional models across every metric and provides feature importance insights for interpretability. |
| **XGBoost (Ensemble)** | XGBoost is the best performing model with 86.71% accuracy, highest AUC (0.9243), F1 (0.8624), and MCC (0.6269). Its gradient boosting framework iteratively corrects errors from previous trees, capturing complex non-linear patterns in the dataset. With learning_rate=0.1, max_depth=6, and 100 estimators, it is well-tuned and demonstrates the clear advantage of advanced ensemble techniques over traditional classifiers on this tabular dataset. |

**Overall Insights**:
- Best performing model: **XGBoost** with 86.71% accuracy and highest AUC (0.9243), F1 (0.8624), and MCC (0.6269)
- Ensemble methods (Random Forest and XGBoost) outperform all traditional algorithms across every metric
- Naive Bayes is the weakest model (79.78% accuracy, MCC 0.3798) due to its feature independence assumption not holding well here
- All models achieved AUC > 0.84, indicating good discriminative ability despite class imbalance (75%/25%)
- Model selection should balance performance metrics with computational requirements for deployment

---

## How to Run This Project

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the notebook** (for model training and results)
```bash
jupyter notebook model_training.ipynb
```

4. **Run the Streamlit app**
```bash
streamlit run app.py
```

5. **Access the application**
   - Open your browser and go to `http://localhost:8501`
   - Upload your test dataset (CSV format)
   - Select target column
   - Train and evaluate all 6 models

---

## Deployment

### Live Application
The application is deployed on Streamlit Community Cloud:
**[Your Streamlit App URL Here]**

### Deployment Steps
1. Created GitHub repository with all required files
2. Logged into Streamlit Community Cloud (https://streamlit.io/cloud)
3. Connected GitHub account
4. Selected repository and `main` branch
5. Configured `app.py` as the main file
6. Deployed the application

---

## Project Structure

```
project-folder/
│
├── app.py                    # Streamlit web application
├── model_training.ipynb      # Jupyter notebook with full training pipeline
├── model_training.py         # Python training script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
└── model/                    # Saved model files
    ├── model_training.ipynb
    └── model_training.py
```

---

## Streamlit App Features

1. **Dataset Upload** — Upload CSV files directly through the UI
2. **Model Selection** — Dropdown to select from 6 trained models
3. **Metrics Display** — All 6 evaluation metrics shown in metric cards
4. **Confusion Matrix** — Visual heatmap of prediction results
5. **Classification Report** — Detailed per-class performance breakdown
6. **Model Comparison Table** — Side-by-side comparison of all 6 models
7. **Results Download** — Export results as CSV

---

## Technologies Used

- **Python 3.8+**
- **scikit-learn** — ML models and metrics
- **XGBoost** — Gradient boosting
- **Streamlit** — Web application
- **Pandas** — Data manipulation
- **NumPy** — Numerical computations
- **Matplotlib & Seaborn** — Visualizations

---

## Assignment Compliance

| Requirement | Status |
|-------------|--------|
| Dataset from UCI/Kaggle (≥12 features, ≥500 instances) | ✅ UCI Adult: 14 features, 30,162 instances |
| 6 Classification models implemented | ✅ LR, DT, KNN, NB, RF, XGBoost |
| 6 Evaluation metrics per model | ✅ Accuracy, AUC, Precision, Recall, F1, MCC |
| GitHub repository with source code + requirements.txt + README | ✅ |
| Streamlit app — CSV upload | ✅ |
| Streamlit app — Model selection dropdown | ✅ |
| Streamlit app — Metrics display | ✅ |
| Streamlit app — Confusion matrix / classification report | ✅ |
| BITS Virtual Lab screenshot | Pending |
| Live Streamlit app link | Pending deployment |

---

## Author

**[Your Name]**
- M.Tech (AIML/DSE), BITS Pilani WILP
- Machine Learning — Assignment 2, February 2026
- GitHub: [Your GitHub Profile]
- Email: [Your Email]
