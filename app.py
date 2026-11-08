
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Subscription Prediction")
st.write(
    "Machine Learning classification project using five trained models "
    "to predict whether a customer subscribes to a term deposit."
)

MODEL_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

# Model comparison
st.header("📊 Model Comparison")

if os.path.exists("model_results.csv"):
    results = pd.read_csv("model_results.csv")
    st.dataframe(results, use_container_width=True)

    st.subheader("Accuracy Comparison")
    st.bar_chart(results.set_index("Model")["Accuracy"])

    st.subheader("AUC Comparison")
    st.bar_chart(results.set_index("Model")["AUC"])
else:
    st.warning("model_results.csv not found. Run train.py first.")

# Upload test data
st.header("📁 Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing features and target column 'y'",
    type=["csv"]
)

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file, sep=None, engine="python")

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(test_data.head(), use_container_width=True)

    if "y" not in test_data.columns:
        st.error("Uploaded dataset must contain the target column 'y'.")
        st.stop()

    X_test = test_data.drop("y", axis=1)
    y_test = test_data["y"].map({"no": 0, "yes": 1})

    if y_test.isna().any():
        st.error("Target column 'y' must contain only 'yes' or 'no'.")
        st.stop()

    st.header("🤖 Select Model")

    selected_model = st.selectbox(
        "Choose a trained model",
        list(MODEL_NAMES.keys())
    )

    model_path = os.path.join(
        "models",
        MODEL_NAMES[selected_model]
    )

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    try:
        y_pred = model.predict(X_test)

        if hasattr(model.named_steps["classifier"], "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.header("📈 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        st.header("📉 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Subscription", "Subscription"]
        )
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        plt.xticks(rotation=20)
        st.pyplot(fig)
        plt.close(fig)

        st.header("📋 Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            target_names=["No Subscription", "Subscription"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)

        st.success(
            f"Evaluation completed successfully using {selected_model}."
        )

    except Exception as e:
        st.error(f"Error during model evaluation: {e}")

else:
    st.info(
        "Upload a test CSV file to evaluate a selected trained model."
    )

st.sidebar.header("Project Information")
st.sidebar.write("Project: Bank Marketing Subscription Prediction")
st.sidebar.write("Dataset: UCI Bank Marketing")
st.sidebar.write("Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM")
st.sidebar.write("Metrics: Accuracy, AUC, Precision, Recall, F1, MCC")