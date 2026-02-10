"""
Machine Learning Assignment 2 - Model Training Script
======================================================

This script implements 6 classification models and evaluates them using 6 metrics.

Models Implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Evaluation Metrics:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC)

Author: Student
Date: February 2026
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


class MLClassificationPipeline:
    """
    A comprehensive ML pipeline for training and evaluating multiple classification models.
    
    Attributes:
        models (dict): Dictionary containing all classification models
        results (dict): Dictionary storing evaluation metrics for each model
        X_train, X_test, y_train, y_test: Train-test split data
        scaler: StandardScaler for feature normalization
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the ML pipeline with all required models.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        
        # Initialize all models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all 6 classification models with optimal hyperparameters."""
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10,
                min_samples_split=5
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                metric='euclidean'
            ),
            'Naive Bayes': GaussianNB(),
            'Random Forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=15,
                min_samples_split=5
            ),
            'XGBoost': XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss'
            )
        }
    
    def load_and_preprocess_data(self, file_path, target_column):
        """
        Load dataset and perform preprocessing.
        
        Args:
            file_path (str): Path to the CSV dataset
            target_column (str): Name of the target variable column
            
        Returns:
            tuple: Preprocessed X and y data
        """
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Features: {df.shape[1] - 1}")
        print(f"Instances: {df.shape[0]}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables in features
        X = self._encode_categorical_features(X)
        
        # Encode target variable if needed
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            print(f"Target classes: {self.label_encoder.classes_}")
        
        print(f"\nPreprocessing complete!")
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return X, y
    
    def _encode_categorical_features(self, X):
        """
        Encode categorical features using Label Encoding.
        
        Args:
            X (DataFrame): Feature matrix
            
        Returns:
            DataFrame: Encoded feature matrix
        """
        X_encoded = X.copy()
        categorical_columns = X_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
        
        return X_encoded
    
    def split_and_scale_data(self, X, y, test_size=0.2):
        """
        Split data into train-test sets and apply feature scaling.
        
        Args:
            X (array-like): Feature matrix
            y (array-like): Target vector
            test_size (float): Proportion of test set
        """
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nData split complete:")
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate all 6 evaluation metrics for a model.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_pred_proba (array): Predicted probabilities (for AUC)
            
        Returns:
            dict: Dictionary containing all metrics
        """
        metrics = {}
        
        # Determine if binary or multi-class
        n_classes = len(np.unique(y_true))
        is_binary = n_classes == 2
        
        # 1. Accuracy
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        
        # 2. AUC Score
        try:
            if is_binary:
                if y_pred_proba is not None:
                    # For binary classification, use probabilities of positive class
                    if len(y_pred_proba.shape) > 1:
                        metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
                else:
                    metrics['AUC'] = roc_auc_score(y_true, y_pred)
            else:
                # Multi-class: use OVR strategy
                if y_pred_proba is not None:
                    metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, 
                                                   multi_class='ovr', average='weighted')
                else:
                    metrics['AUC'] = 0.0  # Cannot calculate without probabilities
        except Exception as e:
            print(f"AUC calculation warning: {e}")
            metrics['AUC'] = 0.0
        
        # 3. Precision
        metrics['Precision'] = precision_score(y_true, y_pred, 
                                               average='weighted', zero_division=0)
        
        # 4. Recall
        metrics['Recall'] = recall_score(y_true, y_pred, 
                                        average='weighted', zero_division=0)
        
        # 5. F1 Score
        metrics['F1'] = f1_score(y_true, y_pred, 
                                average='weighted', zero_division=0)
        
        # 6. Matthews Correlation Coefficient
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        
        return metrics
    
    def train_and_evaluate_all_models(self):
        """
        Train all models and evaluate them using all metrics.
        
        Returns:
            dict: Results dictionary with metrics for all models
        """
        print("\n" + "="*70)
        print("TRAINING AND EVALUATING ALL MODELS")
        print("="*70)
        
        for model_name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"Training: {model_name}")
            print(f"{'='*70}")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Get prediction probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(self.X_test)
            
            # Calculate all metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Store results
            self.results[model_name] = metrics
            
            # Print metrics
            print(f"\nPerformance Metrics for {model_name}:")
            print("-" * 50)
            for metric_name, value in metrics.items():
                print(f"{metric_name:.<25} {value:.4f}")
        
        return self.results
    
    def get_results_dataframe(self):
        """
        Convert results to a pandas DataFrame for easy viewing.
        
        Returns:
            DataFrame: Results in tabular format
        """
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.round(4)
        return df_results
    
    def save_models(self, save_dir='models'):
        """
        Save all trained models to disk.
        
        Args:
            save_dir (str): Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = os.path.join(save_dir, f"{model_name.replace(' ', '_').lower()}.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved: {filename}")
        
        # Save scaler
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved: {scaler_path}")
        
        # Save label encoder if exists
        if self.label_encoder is not None:
            le_path = os.path.join(save_dir, 'label_encoder.pkl')
            with open(le_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"Saved: {le_path}")
    
    def get_confusion_matrix(self, model_name):
        """
        Get confusion matrix for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            array: Confusion matrix
        """
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)
    
    def get_classification_report(self, model_name):
        """
        Get classification report for a specific model.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            str: Classification report
        """
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        # Get class names if label encoder exists
        target_names = None
        if self.label_encoder is not None:
            target_names = self.label_encoder.classes_
        
        return classification_report(self.y_test, y_pred, target_names=target_names)


def main():
    """
    Main function to run the complete ML pipeline.
    
    Usage:
        Update DATA_PATH and TARGET_COLUMN variables with your dataset details.
    """
    # ========== CONFIGURATION ==========
    # Update these variables with your dataset details
    DATA_PATH = 'your_dataset.csv'  # Path to your dataset
    TARGET_COLUMN = 'target'  # Name of your target column
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # ===================================
    
    print("="*70)
    print("MACHINE LEARNING CLASSIFICATION PIPELINE")
    print("="*70)
    
    # Initialize pipeline
    pipeline = MLClassificationPipeline(random_state=RANDOM_STATE)
    
    # Load and preprocess data
    X, y = pipeline.load_and_preprocess_data(DATA_PATH, TARGET_COLUMN)
    
    # Split and scale data
    pipeline.split_and_scale_data(X, y, test_size=TEST_SIZE)
    
    # Train and evaluate all models
    results = pipeline.train_and_evaluate_all_models()
    
    # Display results
    print("\n" + "="*70)
    print("FINAL RESULTS - ALL MODELS")
    print("="*70)
    results_df = pipeline.get_results_dataframe()
    print(results_df.to_string())
    
    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    pipeline.save_models()
    
    # Save results to CSV
    results_df.to_csv('model_results.csv')
    print("\nResults saved to: model_results.csv")
    
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
