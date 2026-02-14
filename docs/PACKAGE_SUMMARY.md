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
| `input/wine_quality_red.csv` | Wine Quality Red dataset |

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

**Wine Quality Red Dataset** — loaded from `input/wine_quality_red.csv`.
- Source: https://archive.ics.uci.edu/ml/datasets/wine+quality
- 1,599 instances | 12 features | Multi-class classification (quality score)

---

## Model Results (already computed)

| Model | Accuracy | AUC | F1 | MCC |
|-------|----------|-----|----|-----|
| Logistic Regression | 0.5906 | 0.7555 | 0.5673 | 0.3250 |
| Decision Tree | 0.5938 | 0.7080 | 0.5921 | 0.3639 |
| K-Nearest Neighbors | 0.6094 | 0.7476 | 0.5959 | 0.3733 |
| Naive Bayes | 0.5625 | 0.7377 | 0.5681 | 0.3299 |
| Random Forest | 0.6625 | 0.8338 | 0.6462 | 0.4547 |
| XGBoost | **0.6781** | **0.8171** | **0.6687** | **0.4867** |

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
| BITS Lab screenshot | 1 | Done — Streamlit app tested with wine_quality_red.csv on BITS Virtual Lab |
