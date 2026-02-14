# START HERE - ML Assignment 2

## What You Have

A complete, ready-to-run ML classification project using the **UCI Adult Income dataset** loaded from the local `input/adult_income.csv` file.

---

## Project Structure

```
mlassignment2/
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
├── README.md                     # GitHub documentation (complete)
├── .gitignore                    # Git ignore rules
│
├── model/
│   └── model_training.ipynb      # Jupyter notebook (run this on BITS Lab)
│
├── input/                        # Dataset files
│   └── adult_income.csv          # UCI Adult Income dataset
│
├── output/                       # Generated after running notebook
│   ├── model_comparison_results.csv
│   ├── confusion_matrices.png
│   └── metrics_comparison.png
│
├── utils/                        # Shared ML utilities
│   ├── __init__.py
│   └── ml_utils.py
│
└── docs/                         # This folder
    ├── START_HERE.md
    ├── PACKAGE_SUMMARY.md
    ├── DEPLOYMENT_GUIDE.md
    └── SUBMISSION_TEMPLATE.md
```

---

## Dataset

**UCI Adult Income** — loaded from `input/adult_income.csv`.
- 30,162 instances | 14 features | Binary classification
- Target: `income` (<=50K or >50K)

---

## What You Need to Do

### 1. Run on BITS Virtual Lab
- Upload `model/model_training.ipynb`
- Run all cells (Kernel → Restart & Run All)
- **Take a screenshot** of the results table (Section 9 output)

### 2. Push to GitHub
- Create a new **public** repository
- Push these files: `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `model/`, `input/`, `output/`, `utils/`, `docs/`

### 3. Deploy to Streamlit Cloud
- Go to https://streamlit.io/cloud
- Connect GitHub → select repo → select `app.py` → Deploy

### 4. Fill in README.md
- Add your name, GitHub link, Streamlit app URL
- Metric values are already filled in

### 5. Create and Submit PDF
- Use `docs/SUBMISSION_TEMPLATE.md` as your guide
- Include: GitHub link, Streamlit link, BITS screenshot, README content
- Submit on Taxila — click **SUBMIT**, not "Save Draft"

---

## Deadline: 15-Feb-2026, 23:59 PM

**For BITS Lab issues:** neha.vinayak@pilani.bits-pilani.ac.in
Subject: "ML Assignment 2: BITS Lab issue"
