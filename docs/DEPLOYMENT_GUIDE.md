# Deployment and Submission Guide
## ML Assignment 2 - Step by Step

---

## Phase 1: BITS Virtual Lab Execution

- [ ] Log into BITS Virtual Lab
- [ ] Upload `model/model_training.ipynb`
- [ ] Open the notebook in Jupyter
- [ ] Run all cells: **Kernel → Restart & Run All**
  - Dataset loads automatically from UCI URL (no CSV upload needed)
  - All 6 models train sequentially
  - Results table appears in cell-17 output
- [ ] **Take a screenshot** showing:
  - BITS Virtual Lab interface visible
  - Results table with all 6 models and 6 metrics
- [ ] Output files are saved to `output/` folder automatically

---

## Phase 2: GitHub Repository Setup

- [ ] Go to https://github.com → New Repository
  - Name: `ml-assignment-2` (or similar)
  - Visibility: **Public** (required for Streamlit)
- [ ] Clone to your machine:
  ```bash
  git clone https://github.com/yourusername/ml-assignment-2.git
  ```
- [ ] Copy all project files into the repo:
  ```
  app.py
  requirements.txt
  README.md
  .gitignore
  model/model_training.ipynb
  model/model_training.py
  output/model_comparison_results.csv
  output/confusion_matrices.png
  output/metrics_comparison.png
  docs/
  ```
- [ ] Update `README.md`:
  - Add your name, email, GitHub profile
  - Add Streamlit app URL (after deployment)
  - All metric values are already filled in
- [ ] Commit and push:
  ```bash
  git add .
  git commit -m "ML Assignment 2: Classification Models"
  git push origin main
  ```
- [ ] Verify repo is Public and all files are visible

---

## Phase 3: Streamlit App Deployment

- [ ] Go to https://streamlit.io/cloud
- [ ] Sign in with GitHub
- [ ] Click **New app**
  - Repository: your repo
  - Branch: `main`
  - Main file: `app.py`
- [ ] Click **Deploy**
- [ ] Wait 3-5 minutes for deployment
- [ ] Test the live app:
  - [ ] CSV upload works
  - [ ] Model selection dropdown appears after training
  - [ ] All 6 metrics display
  - [ ] Confusion matrix renders
  - [ ] Classification report shows
- [ ] Copy the app URL: `https://your-app.streamlit.app`
- [ ] Add the URL to `README.md` and push again

**Common Issues:**
```
ModuleNotFoundError → check requirements.txt has xgboost
App crashes         → test locally: streamlit run app.py
Repo not found      → ensure repository is Public
```

---

## Phase 4: Submission PDF

- [ ] Open `docs/SUBMISSION_TEMPLATE.md`
- [ ] Copy content to Word / Google Docs
- [ ] Fill in:
  - Your name, student ID, email
  - GitHub repository URL
  - Streamlit app URL
  - Insert BITS Lab screenshot in Section 3
  - Paste complete README.md content in Section 4
- [ ] Verify no `[...]` placeholders remain
- [ ] Export as PDF
- [ ] Check PDF: links clickable, screenshot visible

---

## Phase 5: Submit on Taxila

- [ ] Log into Taxila portal
- [ ] Navigate to ML Assignment 2
- [ ] Upload your PDF
- [ ] Click **SUBMIT** (NOT "Save Draft")
- [ ] Confirm submission status shows "Submitted"
- [ ] Save confirmation before deadline

**Deadline: 15-Feb-2026, 23:59 PM**

---

## Grading Breakdown (15 Marks)

| Requirement | Marks |
|-------------|-------|
| All 6 models implemented + all 6 metrics | 6 |
| Dataset description in README | 1 |
| Model observations | 3 |
| Streamlit: CSV upload | 1 |
| Streamlit: model dropdown | 1 |
| Streamlit: metrics display | 1 |
| Streamlit: confusion matrix | 1 |
| BITS Lab screenshot | 1 |
| **Total** | **15** |

---

## Contact

**BITS Lab issues:** neha.vinayak@pilani.bits-pilani.ac.in
Subject: "ML Assignment 2: BITS Lab issue"
