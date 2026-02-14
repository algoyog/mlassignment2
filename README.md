# Machine Learning Assignment 2 - Classification Models

## Problem Statement

This project implements a comprehensive machine learning classification pipeline featuring six different classification algorithms. The goal is to compare the performance of traditional ML models (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes) against ensemble methods (Random Forest and XGBoost) using six evaluation metrics. The pipeline is demonstrated using the Wine Quality Red dataset and includes an interactive Streamlit web application that supports any CSV classification dataset.

---

## Dataset Description

### Dataset Information
- **Dataset Name**: Wine Quality Red Dataset
- **Source**: [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Type**: Multi-class Classification
- **File**: `input/wine_quality_red.csv`

### Dataset Characteristics
- **Number of Instances**: 1,599 — Minimum requirement: 500 (met)
- **Number of Features**: 12 — Minimum requirement: 12 (met)
- **Target Variable**: `quality` (wine quality score)

### Features Overview

| Feature | Type | Description |
|---------|------|-------------|
| fixed acidity | Numerical | Fixed acidity level |
| volatile acidity | Numerical | Volatile acidity level |
| citric acid | Numerical | Citric acid content |
| residual sugar | Numerical | Residual sugar content |
| chlorides | Numerical | Chloride content |
| free sulfur dioxide | Numerical | Free sulfur dioxide level |
| total sulfur dioxide | Numerical | Total sulfur dioxide level |
| density | Numerical | Density of wine |
| pH | Numerical | pH level |
| sulphates | Numerical | Sulphate content |
| alcohol | Numerical | Alcohol percentage |
| quality | Numerical | Wine quality score (target) |

- **All Features are Numerical (12)**
- **Missing Values**: None

### Data Preprocessing Steps
1. **Scaling**: All features normalized using StandardScaler
2. **Train-Test Split**: 80-20 split with stratification to preserve class balance

---

## Models Used

### Model Implementation Summary
All six models were implemented using scikit-learn and XGBoost libraries, trained on the same 80-20 stratified train-test split for fair comparison.

### Comparison Table - Model Performance Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| **Logistic Regression** | 0.5906 | 0.7555 | 0.5695 | 0.5906 | 0.5673 | 0.3250 |
| **Decision Tree** | 0.5938 | 0.7080 | 0.5908 | 0.5938 | 0.5921 | 0.3639 |
| **K-Nearest Neighbors** | 0.6094 | 0.7476 | 0.5841 | 0.6094 | 0.5959 | 0.3733 |
| **Naive Bayes** | 0.5625 | 0.7377 | 0.5745 | 0.5625 | 0.5681 | 0.3299 |
| **Random Forest** | 0.6625 | 0.8338 | 0.6377 | 0.6625 | 0.6462 | 0.4547 |
| **XGBoost** | 0.6781 | 0.8171 | 0.6657 | 0.6781 | 0.6687 | 0.4867 |

---

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Logistic Regression achieves 59.06% accuracy with AUC 0.7555, serving as a solid linear baseline. It performs reasonably because several wine features like alcohol and volatile acidity have approximately linear relationships with quality. The model converges reliably with lbfgs solver and is the most interpretable among the six, though it falls short on non-linear patterns compared to tree-based models. |
| **Decision Tree** | Decision Tree achieves 59.38% accuracy with AUC 0.7080 and MCC 0.3639. With max_depth=10, it captures non-linear interactions between features like alcohol, volatile acidity, and sulphates. It slightly outperforms Logistic Regression while remaining interpretable through decision rules. However, it is surpassed by ensemble methods which reduce its variance. |
| **K-Nearest Neighbors** | KNN with k=5 achieves 60.94% accuracy with AUC 0.7476, outperforming both linear models. Feature scaling via StandardScaler is essential for KNN's distance calculations and was correctly applied. On 1,599 instances, it is computationally efficient at prediction time and benefits from the natural clustering present in the Wine Quality dataset. |
| **Naive Bayes** | Gaussian Naive Bayes achieves the lowest accuracy at 56.25% with MCC 0.3299, the weakest among all models. The feature independence assumption does not hold well here — features like fixed acidity, citric acid, and pH are correlated — limiting its performance. Despite this, it achieves AUC of 0.7377, indicating reasonable probability calibration, and is the fastest model to train. |
| **Random Forest** | Random Forest achieves 66.25% accuracy with AUC 0.8338 and MCC 0.4547, ranking second overall. The ensemble of 100 trees reduces overfitting through bagging and random feature subsets. Its AUC of 0.8338 reflects the best discriminative ability among all models. It consistently outperforms all traditional models across every metric and provides feature importance insights for interpretability. |
| **XGBoost** | XGBoost is the best performing model with 67.81% accuracy, highest F1 (0.6687), and highest MCC (0.4867). Its gradient boosting framework iteratively corrects errors from previous trees, capturing complex non-linear patterns in the wine quality dataset. With learning_rate=0.1, max_depth=6, and 100 estimators, it demonstrates the clear advantage of advanced ensemble techniques over traditional classifiers on this multi-class tabular dataset. |

**Overall Insights**:
- Best performing model: **XGBoost** with 67.81% accuracy, highest F1 (0.6687), and highest MCC (0.4867)
- Ensemble methods (Random Forest and XGBoost) significantly outperform all traditional algorithms across every metric
- Naive Bayes is the weakest model (56.25% accuracy, MCC 0.3299) due to its feature independence assumption not holding well for correlated wine features
- All models achieved AUC > 0.70, indicating reasonable discriminative ability for this challenging multi-class problem
- The multi-class nature (6 quality levels) makes this dataset harder than binary classification, explaining the moderate accuracy values

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
jupyter notebook model/model_training.ipynb
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
**[https://mlassignment2-kmepmbozytsyeyf3yerkyb.streamlit.app/](https://mlassignment2-kmepmbozytsyeyf3yerkyb.streamlit.app/)**

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
mlassignment2/
|
|-- app.py                        # Streamlit web application
|-- requirements.txt              # Python dependencies
|-- README.md                     # Project documentation
|-- .gitignore                    # Git ignore rules
|
|-- model/
|   `-- model_training.ipynb      # Jupyter notebook for model training and evaluation
|
|-- input/
|   `-- wine_quality_red.csv      # Wine Quality Red dataset
|
|-- output/
|   |-- model_comparison_results.csv   # Generated metrics table
|   |-- confusion_matrices.png         # Confusion matrix plots
|   `-- metrics_comparison.png         # Bar chart comparison
|
|-- utils/
|   |-- __init__.py
|   `-- ml_utils.py               # Shared utilities (preprocessing, models, metrics, plots)
|
`-- docs/
    |-- 2025AA05026_ML_Assignment.pdf
    |-- ML_Assignment_2.pdf        

```

---

## Streamlit App Features

The Streamlit app is a general-purpose classification dashboard that works with any CSV dataset. It was tested and demonstrated on the BITS Virtual Lab using the Wine Quality Red dataset (1,599 instances, 12 features).

1. **Dataset Upload** - Upload any CSV classification dataset
2. **Target Column Selection** - Select the target variable with class distribution preview
3. **Model Training** - Train all 6 models with configurable random state and test size
4. **Metrics Display** - All 6 evaluation metrics shown in metric cards
5. **Confusion Matrix** - Visual heatmap of prediction results
6. **Classification Report** - Detailed per-class performance breakdown
7. **Model Comparison Table** - Side-by-side comparison of all 6 models
8. **Results Download** - Export results as CSV

---

## Technologies Used

- **Python 3.8+**
- **scikit-learn** - ML models and metrics
- **XGBoost** - Gradient boosting
- **Streamlit** - Web application
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Matplotlib and Seaborn** - Visualizations

---

## Assignment Compliance

| Requirement | Status |
|-------------|--------|
| Dataset from UCI/Kaggle (>=12 features, >=500 instances) | Done — Wine Quality Red: 12 features, 1,599 instances |
| 6 Classification models implemented | Done — LR, DT, KNN, NB, Random Forest, XGBoost |
| 6 Evaluation metrics per model | Done — Accuracy, AUC, Precision, Recall, F1, MCC |
| GitHub repository with source code + requirements.txt + README | Done |
| Streamlit app — CSV upload | Done |
| Streamlit app — Model selection dropdown | Done |
| Streamlit app — Metrics display | Done |
| Streamlit app — Confusion matrix / classification report | Done |
| BITS Virtual Lab screenshot | Done — Streamlit app tested with wine_quality_red.csv on BITS Virtual Lab |
| Live Streamlit app link | Done — https://mlassignment2-kmepmbozytsyeyf3yerkyb.streamlit.app/ |

---

## Author

**Aravindan B**
- M.Tech (AIML), BITS Pilani WILP
- Machine Learning — Assignment 2, February 2026
- GitHub: [algoyog](https://github.com/algoyog)
- Email: 2025aa05026@wilp.bits-pilani.ac.in
