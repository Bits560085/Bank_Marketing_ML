
import os
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

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

# Create models directory
os.makedirs("models", exist_ok=True)

# Load dataset
print("Loading Bank Marketing dataset...")
df = pd.read_csv("data/bank-full.csv", sep=";")

print("Dataset Shape:", df.shape)
print(df.head())

# Target variable
X = df.drop("y", axis=1)
y = df["y"].map({"no": 0, "yes": 1})

# Identify feature types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Preprocessing pipelines
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="linear")
}

results = []

# Train and evaluate
for name, classifier in models.items():
    print("\n" + "=" * 60)
    print("Training:", name)

    start_time = time.time()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    # AUC using prediction scores
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        y_score = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_score = pipeline.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    elapsed = time.time() - start_time

    print("Accuracy:", round(accuracy, 4))
    print("AUC:", round(auc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("MCC:", round(mcc, 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Training and evaluation time:", round(elapsed, 2), "seconds")

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc,
        "Time (seconds)": elapsed
    })

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipeline, os.path.join("models", filename))
    print("Saved model:", filename)

# Save comparison results
results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print(results_df.round(4).to_string(index=False))
print("\nResults saved to model_results.csv")
print("All models saved in the models folder.")