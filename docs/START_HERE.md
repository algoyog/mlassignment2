# START HERE - ML Assignment 2

## What You Have

A complete, ready-to-run ML classification project using the **Wine Quality Red dataset** loaded from the local `input/wine_quality_red.csv` file.

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
│   └── wine_quality_red.csv      # Wine Quality Red dataset
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

**Wine Quality Red** — loaded from `input/wine_quality_red.csv`.
- 1,599 instances | 12 features | Multi-class classification
- Target: `quality` (wine quality score)

---

## What You Need to Do

### 1. Test on BITS Virtual Lab
- Open the deployed Streamlit app on the BITS Virtual Lab browser
- Upload a test dataset (e.g., `input/wine_quality_red.csv`)
- Train all 6 models and view the Model Comparison Table
- **Take screenshots** showing the BITS Virtual Lab interface with the app running

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
