"""
ml_utils.py - Shared ML utilities for model training and evaluation.
Used by both app.py and model/model_training.ipynb.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from xgboost import XGBClassifier


def preprocess_data(df, target_column):
    """
    Preprocess dataset: encode categoricals, encode target.

    Returns:
        tuple: (X, y, label_encoder)
    """
    df_processed = df.copy()
    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]

    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Always encode target to ensure 0-indexed classes (required by XGBoost)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, label_encoder


def initialize_models(random_state=42):
    """
    Initialize all 6 classification models.

    Returns:
        dict: Model name -> model object
    """
    return {
        'Logistic Regression': LogisticRegression(
            random_state=random_state, max_iter=1000, solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=random_state, max_depth=10, min_samples_split=5
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5, weights='uniform'
        ),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(
            random_state=random_state, n_estimators=100, max_depth=15
        ),
        'XGBoost': XGBClassifier(
            random_state=random_state, n_estimators=100,
            max_depth=6, learning_rate=0.1, eval_metric='logloss'
        )
    }


def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Calculate all 6 evaluation metrics.

    Returns:
        dict: metric name -> value
    """
    metrics = {}
    is_binary = len(np.unique(y_true)) == 2

    metrics['Accuracy'] = accuracy_score(y_true, y_pred)

    try:
        if is_binary:
            if y_pred_proba is not None and len(y_pred_proba.shape) > 1:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred)
        else:
            if y_pred_proba is not None:
                metrics['AUC'] = roc_auc_score(
                    y_true, y_pred_proba, multi_class='ovr', average='weighted'
                )
            else:
                metrics['AUC'] = 0.0
    except Exception:
        metrics['AUC'] = 0.0

    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['Recall']    = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1']        = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['MCC']       = matthews_corrcoef(y_true, y_pred)

    return metrics


def plot_confusion_matrix(cm, class_names=None):
    """
    Plot confusion matrix as a seaborn heatmap.

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names if class_names is not None else 'auto',
        yticklabels=class_names if class_names is not None else 'auto',
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
