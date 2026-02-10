import streamlit as st
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from utils.ml_utils import preprocess_data, initialize_models, calculate_all_metrics, plot_confusion_matrix

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ML Classification Models",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def display_metric_cards(metrics):
    cols = st.columns(3)
    metric_list = list(metrics.items())
    for idx, (metric_name, value) in enumerate(metric_list):
        col_idx = idx % 3
        with cols[col_idx]:
            st.metric(label=metric_name, value=f"{value:.4f}", delta=None)


def main():
    st.markdown('<h1 class="main-header">🤖 ML Classification Models Dashboard</h1>',
                unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")

        random_state = st.number_input(
            "Random State",
            min_value=0,
            max_value=100,
            value=42,
            help="Set random state for reproducibility"
        )

        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data for testing"
        )

        st.markdown("---")
        st.markdown("""
        ### 📊 Models Implemented
        1. Logistic Regression
        2. Decision Tree
        3. K-Nearest Neighbors
        4. Naive Bayes
        5. Random Forest
        6. XGBoost

        ### 📈 Metrics Calculated
        - Accuracy
        - AUC Score
        - Precision
        - Recall
        - F1 Score
        - MCC Score
        """)

    st.markdown("## 📁 Step 1: Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload your CSV dataset (test data recommended for free tier)",
        type=['csv'],
        help="Upload a CSV file with your test data"
    )

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        if df is not None:
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")

            with st.expander("📊 View Dataset Preview"):
                st.dataframe(df.head(10), use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Dataset Info:**")
                    st.write(f"- Rows: {df.shape[0]}")
                    st.write(f"- Columns: {df.shape[1]}")

                with col2:
                    st.write("**Column Types:**")
                    st.write(df.dtypes.value_counts())

            st.markdown("---")
            st.markdown("## 🎯 Step 2: Select Target Column")

            target_column = st.selectbox(
                "Select the target variable (column to predict)",
                options=df.columns.tolist(),
                help="Choose the column you want to predict"
            )

            if target_column:
                st.info(f"🎯 Target column: **{target_column}**")

                if df[target_column].dtype == 'object' or df[target_column].nunique() < 20:
                    st.write("**Class Distribution:**")
                    class_dist = df[target_column].value_counts()
                    st.bar_chart(class_dist)

                st.markdown("---")
                st.markdown("## 🤖 Step 3: Train Models")

                if st.button("🚀 Train All Models", type="primary"):
                    with st.spinner("Training models... This may take a moment..."):
                        try:
                            X, y, label_encoder = preprocess_data(df, target_column)

                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size,
                                random_state=random_state, stratify=y
                            )

                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)

                            st.success(f"✅ Data preprocessed! Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

                            models = initialize_models(random_state)
                            results = {}
                            trained_models = {}

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for idx, (model_name, model) in enumerate(models.items()):
                                status_text.text(f"Training {model_name}...")
                                model.fit(X_train_scaled, y_train)
                                y_pred = model.predict(X_test_scaled)
                                y_pred_proba = None
                                if hasattr(model, 'predict_proba'):
                                    y_pred_proba = model.predict_proba(X_test_scaled)
                                metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)
                                results[model_name] = metrics
                                trained_models[model_name] = {
                                    'model': model,
                                    'y_pred': y_pred,
                                    'y_true': y_test
                                }
                                progress_bar.progress((idx + 1) / len(models))

                            status_text.text("Training complete!")
                            st.success("✅ All models trained successfully!")

                            st.session_state['results'] = results
                            st.session_state['trained_models'] = trained_models
                            st.session_state['label_encoder'] = label_encoder
                            st.session_state['y_test'] = y_test

                        except Exception as e:
                            st.error(f"❌ Error during training: {str(e)}")
                            st.exception(e)

                if 'results' in st.session_state:
                    st.markdown("---")
                    st.markdown("## 📊 Step 4: View Results")

                    st.markdown("### 🔍 Select Model to Analyze")
                    selected_model = st.selectbox(
                        "Choose a model",
                        options=list(st.session_state['results'].keys()),
                        help="Select a model to view detailed metrics"
                    )

                    if selected_model:
                        st.markdown(f"### 📈 Results for: **{selected_model}**")

                        st.markdown("#### Evaluation Metrics")
                        metrics = st.session_state['results'][selected_model]
                        display_metric_cards(metrics)

                        st.markdown("#### Detailed Metrics Table")
                        metrics_df = pd.DataFrame([metrics])
                        st.dataframe(
                            metrics_df.style.format("{:.4f}"),
                            use_container_width=True
                        )

                        st.markdown("#### Confusion Matrix")
                        model_data = st.session_state['trained_models'][selected_model]
                        y_true = model_data['y_true']
                        y_pred = model_data['y_pred']

                        cm = confusion_matrix(y_true, y_pred)

                        class_names = None
                        if st.session_state['label_encoder'] is not None:
                            class_names = st.session_state['label_encoder'].classes_

                        fig = plot_confusion_matrix(cm, class_names)
                        st.pyplot(fig)

                        st.markdown("#### Classification Report")
                        report = classification_report(
                            y_true, y_pred,
                            target_names=class_names,
                            output_dict=True
                        )
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(
                            report_df.style.format("{:.4f}"),
                            use_container_width=True
                        )

                    st.markdown("---")
                    st.markdown("### 📊 Model Comparison Table")

                    results_df = pd.DataFrame(st.session_state['results']).T
                    results_df = results_df.round(4)

                    st.dataframe(
                        results_df.style.highlight_max(axis=0, color='lightgreen'),
                        use_container_width=True
                    )

                    best_model = results_df['Accuracy'].idxmax()
                    best_accuracy = results_df['Accuracy'].max()

                    st.success(f"🏆 **Best Model:** {best_model} (Accuracy: {best_accuracy:.4f})")

                    st.markdown("---")
                    st.markdown("### 💾 Download Results")

                    csv = results_df.to_csv()
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="model_comparison_results.csv",
                        mime="text/csv"
                    )

    else:
        st.info("👆 Please upload a CSV file to get started!")

        st.markdown("""
        ### 📝 Instructions:

        1. **Upload Dataset**: Upload your test dataset in CSV format
        2. **Select Target**: Choose the column you want to predict
        3. **Train Models**: Click the button to train all 6 models
        4. **Select Model**: Choose a model from the dropdown to view detailed results
        5. **View Results**: Explore metrics, confusion matrix, and classification report

        ### 📋 Requirements:
        - Minimum 12 features
        - Minimum 500 instances
        - CSV format
        - Clear target column

        ### ⚠️ Note:
        For Streamlit free tier, upload only test data to avoid capacity issues.
        """)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Machine Learning Assignment 2 | M.Tech (AIML/DSE) | 2026</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
