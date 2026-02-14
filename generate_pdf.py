"""Generate submission PDF for ML Assignment 2."""

from fpdf import FPDF
import os

class SubmissionPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, 'ML Assignment 2 - Classification Models | Aravindan B (2025aa05026)', align='C')
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, num, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(25, 60, 120)
        self.cell(0, 10, f'{num}. {title}', new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(25, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_heading(self, text):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(40, 40, 40)
        self.cell(0, 8, text, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_label_value(self, label, value):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(50, 50, 50)
        label_w = self.get_string_width(label) + 2
        self.cell(label_w, 6, label)
        self.set_font('Helvetica', '', 10)
        self.cell(0, 6, value, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    def url_box(self, url):
        self.set_fill_color(240, 245, 250)
        self.set_draw_color(180, 200, 220)
        self.set_font('Courier', '', 9)
        self.set_text_color(25, 60, 120)
        self.cell(0, 8, f'  {url}', border=1, fill=True, new_x='LMARGIN', new_y='NEXT')
        self.ln(3)
        self.set_text_color(50, 50, 50)

    def bullet(self, text, bold_prefix=''):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(50, 50, 50)
        x = self.get_x()
        self.cell(5, 5.5, '-')
        if bold_prefix:
            self.set_font('Helvetica', 'B', 10)
            bp_w = self.get_string_width(bold_prefix) + 1
            self.cell(bp_w, 5.5, bold_prefix)
            self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Header row
        self.set_font('Helvetica', 'B', 8)
        self.set_fill_color(25, 60, 120)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align='C')
        self.ln()

        # Data rows
        self.set_font('Helvetica', '', 8)
        self.set_text_color(40, 40, 40)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(245, 248, 252)
            else:
                self.set_fill_color(255, 255, 255)

            # Highlight best row (XGBoost)
            is_best = row[0].strip() == 'XGBoost'
            if is_best:
                self.set_fill_color(220, 245, 220)
                self.set_font('Helvetica', 'B', 8)

            for i, val in enumerate(row):
                align = 'L' if i == 0 else 'C'
                self.cell(col_widths[i], 6.5, val, border=1, fill=True, align=align)
            self.ln()

            if is_best:
                self.set_font('Helvetica', '', 8)

        self.ln(3)

    def add_observation_table(self, observations):
        col_widths = [40, 150]
        # Header
        self.set_font('Helvetica', 'B', 8)
        self.set_fill_color(25, 60, 120)
        self.set_text_color(255, 255, 255)
        self.cell(col_widths[0], 7, 'ML Model Name', border=1, fill=True, align='C')
        self.cell(col_widths[1], 7, 'Observation about Model Performance', border=1, fill=True, align='C')
        self.ln()

        # Rows
        self.set_text_color(40, 40, 40)
        for idx, (model, obs) in enumerate(observations):
            if idx % 2 == 0:
                self.set_fill_color(245, 248, 252)
            else:
                self.set_fill_color(255, 255, 255)

            x = self.get_x()
            y = self.get_y()

            # Calculate height needed
            self.set_font('Helvetica', '', 7)
            # Estimate lines needed
            line_w = col_widths[1] - 2
            n_lines = max(1, len(obs) // 75 + 1)
            row_h = max(12, n_lines * 3.5)

            # Check if we need a new page
            if y + row_h > 270:
                self.add_page()
                y = self.get_y()

            # Model name cell
            self.set_font('Helvetica', 'B', 8)
            self.set_xy(x, y)
            self.cell(col_widths[0], row_h, model, border=1, fill=True, align='C')

            # Observation cell
            self.set_font('Helvetica', '', 7)
            self.set_xy(x + col_widths[0], y)
            self.multi_cell(col_widths[1], 3.5, obs, border=1, fill=True)

            actual_h = self.get_y() - y
            if actual_h > row_h:
                # Redraw model name cell with correct height
                self.set_xy(x, y)
                self.set_font('Helvetica', 'B', 8)
                self.cell(col_widths[0], actual_h, model, border=1, fill=True, align='C')
                self.set_y(y + actual_h)
            else:
                self.set_y(y + row_h)

        self.ln(3)


def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    SCREENSHOTS = [
        os.path.join('C:/Users/sbara/Desktop', f'{i}.png') for i in [5, 4, 3, 2, 1]
    ]
    SCREENSHOT_CAPTIONS = [
        'Step 1: CSV Upload - wine_quality_red.csv loaded successfully (1599 rows, 12 features)',
        'Step 2: Dataset Preview - Feature table and dataset information displayed',
        'Step 2: Target Column Selection - Class distribution visualization for "quality"',
        'Step 3: Model Training - All 6 models trained successfully on BITS Virtual Lab',
        'Step 4: Model Comparison Table - All 6 models with all 6 evaluation metrics',
    ]

    pdf = SubmissionPDF('P', 'mm', 'A4')
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=20)

    # ===== COVER PAGE =====
    pdf.add_page()
    pdf.ln(30)
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(0, 12, 'Machine Learning', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.cell(0, 12, 'Assignment 2', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)
    pdf.set_font('Helvetica', '', 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, 'Classification Models', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(8)

    # Divider
    pdf.set_draw_color(25, 60, 120)
    pdf.set_line_width(0.8)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.ln(12)

    # Student details box
    pdf.set_fill_color(240, 245, 252)
    pdf.set_draw_color(25, 60, 120)
    box_y = pdf.get_y()
    pdf.rect(35, box_y, 140, 55, style='DF')

    pdf.set_xy(45, box_y + 6)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(60, 7, 'Student Name:')
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(60, 7, 'Aravindan B', new_x='LMARGIN', new_y='NEXT')

    pdf.set_x(45)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(60, 7, 'Student ID:')
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(60, 7, '2025aa05026', new_x='LMARGIN', new_y='NEXT')

    pdf.set_x(45)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(60, 7, 'Program:')
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(60, 7, 'M.Tech (AIML), BITS Pilani WILP', new_x='LMARGIN', new_y='NEXT')

    pdf.set_x(45)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(60, 7, 'Course:')
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(60, 7, 'Machine Learning', new_x='LMARGIN', new_y='NEXT')

    pdf.set_x(45)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(60, 7, 'Submission Date:')
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(60, 7, 'February 2026', new_x='LMARGIN', new_y='NEXT')

    pdf.set_x(45)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(60, 7, 'Email:')
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(60, 7, '2025aa05026@wilp.bits-pilani.ac.in', new_x='LMARGIN', new_y='NEXT')

    pdf.set_y(box_y + 55 + 15)

    # ===== SECTION 1: GitHub Repository =====
    pdf.add_page()
    pdf.section_title(1, 'GitHub Repository Link')
    pdf.sub_heading('Repository URL:')
    pdf.url_box('https://github.com/algoyog/mlassignment2')
    pdf.ln(2)
    pdf.sub_heading('Repository Contents:')
    repo_files = [
        ('app.py', 'Streamlit web application'),
        ('requirements.txt', 'Python dependencies'),
        ('README.md', 'Complete documentation'),
        ('.gitignore', 'Git ignore rules'),
        ('model/model_training.ipynb', 'Jupyter notebook for model training'),
        ('input/adult_income.csv', 'UCI Adult Income dataset'),
        ('output/', 'Generated results and charts'),
        ('utils/ml_utils.py', 'Shared ML utilities'),
    ]
    for fname, desc in repo_files:
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(5, 5.5, '-')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(pdf.get_string_width(fname) + 2, 5.5, fname)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 5.5, f' -- {desc}', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(1)

    # ===== SECTION 2: Streamlit App =====
    pdf.ln(3)
    pdf.section_title(2, 'Live Streamlit App Link')
    pdf.sub_heading('Application URL:')
    pdf.url_box('https://mlassignment2-kmepmbozytsyeyf3yerkyb.streamlit.app/')
    pdf.ln(2)
    pdf.sub_heading('Implemented Features:')
    pdf.bullet('CSV file upload with dataset preview')
    pdf.bullet('Target column selection with class distribution')
    pdf.bullet('Model selection dropdown (6 classification models)')
    pdf.bullet('All 6 evaluation metrics displayed (Accuracy, AUC, Precision, Recall, F1, MCC)')
    pdf.bullet('Confusion matrix visualization')
    pdf.bullet('Classification report')
    pdf.bullet('Model comparison table with best model highlighted')
    pdf.bullet('Results download as CSV')

    # ===== SECTION 3: BITS Virtual Lab Screenshots =====
    pdf.add_page()
    pdf.section_title(3, 'BITS Virtual Lab Screenshots')
    pdf.body_text('The following screenshots demonstrate the Streamlit application running on the BITS Virtual Lab (argo-rdp.codeargo.net) using the Wine Quality Red dataset (1599 instances, 12 features) as a test dataset.')
    pdf.ln(2)

    for i, (img_path, caption) in enumerate(zip(SCREENSHOTS, SCREENSHOT_CAPTIONS)):
        if not os.path.exists(img_path):
            pdf.body_text(f'[Screenshot not found: {img_path}]')
            continue

        # Check if we need a new page (image + caption need ~90mm)
        if pdf.get_y() > 190:
            pdf.add_page()

        # Caption above image
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(25, 60, 120)
        pdf.cell(0, 6, f'Screenshot {i+1}: {caption}', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(1)

        # Image with border
        img_y = pdf.get_y()
        pdf.set_draw_color(180, 200, 220)
        pdf.image(img_path, x=12, w=186)
        img_bottom = pdf.get_y()
        pdf.rect(11, img_y - 1, 188, img_bottom - img_y + 2)
        pdf.ln(8)

    # ===== SECTION 4: README Content =====
    pdf.add_page()
    pdf.section_title(4, 'README Content')

    # Problem Statement
    pdf.sub_heading('Problem Statement')
    pdf.body_text(
        'This project implements a comprehensive machine learning classification pipeline featuring '
        'six different classification algorithms applied to the UCI Adult Income dataset. The goal is '
        'to predict whether an individual\'s annual income exceeds $50K based on census attributes, '
        'and to compare the performance of traditional ML models (Logistic Regression, Decision Tree, '
        'K-Nearest Neighbors, Naive Bayes) against ensemble methods (Random Forest and XGBoost) '
        'using six evaluation metrics.'
    )

    # Dataset Description
    pdf.sub_heading('Dataset Description')
    pdf.bold_label_value('Dataset Name: ', 'UCI Adult Income Dataset (Census Income)')
    pdf.bold_label_value('Source: ', 'UCI Machine Learning Repository')
    pdf.bold_label_value('Type: ', 'Binary Classification')
    pdf.bold_label_value('Instances: ', '30,162 (Requirement >= 500: MET)')
    pdf.bold_label_value('Features: ', '14 (Requirement >= 12: MET)')
    pdf.bold_label_value('Target: ', 'income (<=50K or >50K)')
    pdf.ln(2)
    pdf.body_text('Class Distribution: Class 0 (<=50K): 22,654 samples (75.1%) | Class 1 (>50K): 7,508 samples (24.9%)')
    pdf.ln(1)
    pdf.body_text('Numerical Features (6): age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week')
    pdf.body_text('Categorical Features (8): workclass, education, marital_status, occupation, relationship, race, sex, native_country')
    pdf.ln(2)
    pdf.sub_heading('Data Preprocessing Steps')
    pdf.bullet('Label Encoding on all 8 categorical features')
    pdf.bullet('Target encoded: <=50K to 0, >50K to 1')
    pdf.bullet('StandardScaler normalization on all features')
    pdf.bullet('80-20 stratified train-test split')

    # Model Comparison Table
    pdf.add_page()
    pdf.sub_heading('Model Performance Comparison Table')
    pdf.ln(2)

    headers = ['ML Model Name', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    col_widths = [48, 22, 22, 24, 22, 22, 22]
    rows = [
        ['Logistic Regression',  '0.8175', '0.8501', '0.8060', '0.8175', '0.8018', '0.4613'],
        ['Decision Tree',        '0.8508', '0.8855', '0.8446', '0.8508', '0.8451', '0.5789'],
        ['K-Nearest Neighbors',  '0.8190', '0.8498', '0.8133', '0.8190', '0.8154', '0.4993'],
        ['Naive Bayes',          '0.7978', '0.8498', '0.7830', '0.7978', '0.7697', '0.3798'],
        ['Random Forest',        '0.8589', '0.9136', '0.8534', '0.8589', '0.8526', '0.6003'],
        ['XGBoost',              '0.8671', '0.9243', '0.8624', '0.8671', '0.8624', '0.6269'],
    ]
    pdf.add_table(headers, rows, col_widths)

    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(34, 120, 34)
    pdf.cell(0, 6, 'Best Model: XGBoost (Accuracy = 0.8671, AUC = 0.9243)', new_x='LMARGIN', new_y='NEXT')
    pdf.set_text_color(50, 50, 50)
    pdf.ln(4)

    # Model Performance Observations
    pdf.sub_heading('Model Performance Observations')
    pdf.ln(2)

    observations = [
        ('Logistic Regression',
         'Achieves 81.75% accuracy with AUC 0.8501, serving as a solid linear baseline. '
         'It performs well because features like education_num, age, and hours_per_week have approximately '
         'linear relationships with income. The model converges reliably with lbfgs solver and is the most '
         'interpretable among the six, though it falls short on non-linear patterns compared to tree-based models.'),
        ('Decision Tree',
         'Achieves 85.08% accuracy with AUC 0.8855 and MCC 0.5789. With max_depth=10, it successfully '
         'captures non-linear interactions between occupation, marital_status, and education without extreme '
         'overfitting. It outperforms linear models significantly while remaining interpretable through decision '
         'rules. However, it is surpassed by ensemble methods which reduce its variance.'),
        ('K-Nearest Neighbors',
         'KNN with k=5 achieves 81.90% accuracy with AUC 0.8498, slightly edging out Logistic Regression. '
         'Feature scaling via StandardScaler is essential for KNN distance calculations and was correctly applied. '
         'On 30,162 instances, it is computationally heavier at prediction time but benefits from the demographic '
         'clustering present in the Adult Income dataset.'),
        ('Naive Bayes',
         'Gaussian Naive Bayes achieves the lowest accuracy at 79.78% with MCC 0.3798. The feature independence '
         'assumption does not hold well here as features like education, occupation, and marital_status are correlated, '
         'limiting its performance. Despite this, it achieves a competitive AUC of 0.8498, indicating reasonable '
         'probability calibration, and is the fastest model to train.'),
        ('Random Forest',
         'Achieves 85.89% accuracy with AUC 0.9136 and MCC 0.6003, ranking second overall. The ensemble of 100 '
         'trees reduces overfitting through bagging and random feature subsets. Its AUC of 0.9136 reflects excellent '
         'discriminative ability. It consistently outperforms all traditional models across every metric and provides '
         'feature importance insights for interpretability.'),
        ('XGBoost',
         'Best performing model with 86.71% accuracy, highest AUC (0.9243), F1 (0.8624), and MCC (0.6269). '
         'Its gradient boosting framework iteratively corrects errors from previous trees, capturing complex '
         'non-linear patterns. With learning_rate=0.1, max_depth=6, and 100 estimators, it demonstrates the clear '
         'advantage of advanced ensemble techniques over traditional classifiers on this tabular dataset.'),
    ]
    pdf.add_observation_table(observations)

    # Overall Insights
    pdf.ln(3)
    pdf.sub_heading('Overall Insights')
    pdf.bullet('Best performing model: XGBoost with 86.71% accuracy, AUC 0.9243, F1 0.8624, MCC 0.6269')
    pdf.bullet('Ensemble methods (Random Forest and XGBoost) outperform all traditional algorithms across every metric')
    pdf.bullet('Naive Bayes is weakest (79.78%) due to its feature independence assumption not holding for this correlated dataset')
    pdf.bullet('All models achieved AUC > 0.84, indicating good discriminative ability despite class imbalance (75%/25%)')

    # Author
    pdf.ln(8)
    pdf.set_draw_color(25, 60, 120)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(0, 6, 'Aravindan B | M.Tech (AIML), BITS Pilani WILP | 2025aa05026@wilp.bits-pilani.ac.in',
             align='C', new_x='LMARGIN', new_y='NEXT')

    pdf.ln(8)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, 'END OF SUBMISSION', align='C')

    # Save
    out_path = os.path.join(BASE, 'output', 'ML_Assignment2_Submission.pdf')
    pdf.output(out_path)
    print(f'PDF generated: {out_path}')
    print(f'Pages: {pdf.pages_count}')


if __name__ == '__main__':
    main()
