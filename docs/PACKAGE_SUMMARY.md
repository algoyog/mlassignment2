# ML Assignment 2 - Package Summary

## Files and Their Purpose

### Core Files (push to GitHub)

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web app — deploy this to Streamlit Cloud |
| `requirements.txt` | Python dependencies for deployment |
| `README.md` | GitHub documentation — metric values already filled in |
| `.gitignore` | Keeps repo clean |

### Model Folder (push to GitHub)

| File | Purpose |
|------|---------|
| `model/model_training.ipynb` | Jupyter notebook — run this on BITS Virtual Lab |

### Input Folder (push to GitHub)

| File | Purpose |
|------|---------|
| `input/adult_income.csv` | UCI Adult Income dataset |

### Utils Folder (push to GitHub)

| File | Purpose |
|------|---------|
| `utils/__init__.py` | Package init |
| `utils/ml_utils.py` | Shared utilities (preprocessing, models, metrics, plots) |

### Output Folder (push to GitHub)

| File | Purpose |
|------|---------|
| `output/model_comparison_results.csv` | Generated metrics table |
| `output/confusion_matrices.png` | Confusion matrix plots |
| `output/metrics_comparison.png` | Bar chart comparison |

### Docs Folder (reference only)

| File | Purpose |
|------|---------|
| `docs/START_HERE.md` | Quick start guide |
| `docs/PACKAGE_SUMMARY.md` | This file |
| `docs/DEPLOYMENT_GUIDE.md` | Step-by-step deployment checklist |
| `docs/SUBMISSION_TEMPLATE.md` | Template for creating submission PDF |

---

## Dataset

**UCI Adult Income Dataset** — loaded from `input/adult_income.csv`.
- Source: https://archive.ics.uci.edu/ml/datasets/adult
- 30,162 instances | 14 features | Binary classification (income <=50K / >50K)

---

## Model Results (already computed)

| Model | Accuracy | AUC | F1 | MCC |
|-------|----------|-----|----|-----|
| Logistic Regression | 0.8175 | 0.8501 | 0.8018 | 0.4613 |
| Decision Tree | 0.8508 | 0.8855 | 0.8451 | 0.5789 |
| K-Nearest Neighbors | 0.8190 | 0.8498 | 0.8154 | 0.4993 |
| Naive Bayes | 0.7978 | 0.8498 | 0.7697 | 0.3798 |
| Random Forest | 0.8589 | 0.9136 | 0.8526 | 0.6003 |
| XGBoost | **0.8671** | **0.9243** | **0.8624** | **0.6269** |

---

## Grading Breakdown (15 Marks)

| Component | Marks | Status |
|-----------|-------|--------|
| All 6 models implemented + all 6 metrics | 6 | ✅ |
| Dataset description in README | 1 | ✅ |
| Model observations in README | 3 | ✅ |
| Streamlit CSV upload | 1 | ✅ |
| Streamlit model dropdown | 1 | ✅ |
| Streamlit metrics display | 1 | ✅ |
| Streamlit confusion matrix | 1 | ✅ |
| BITS Lab screenshot | 1 | Pending — take screenshot when running |
