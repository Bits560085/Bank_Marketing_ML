# Bank Marketing Subscription Prediction

## 1. Project Overview
This project uses machine learning classification algorithms to predict whether a bank customer will subscribe to a term deposit based on marketing campaign data.

The project includes model training, performance evaluation, model comparison, and a Streamlit web application for evaluating uploaded test datasets using pre-trained models.

## 2. Dataset
- **Source:** UCI Machine Learning Repository — Bank Marketing
- **Dataset file:** `bank-full.csv`
- **Instances:** 45,211
- **Columns:** 17 (16 input features and 1 target variable)
- **Target:** `y` — whether the customer subscribed (`yes` or `no`)

## 3. Machine Learning Models
The following five classification algorithms were implemented:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Evaluation Metrics
Each model was evaluated using:
- Accuracy
- AUC (Area Under the ROC Curve)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

## 5. Model Performance

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| KNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Gaussian Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| SVM | 0.8932 | 0.9046 | 0.6503 | 0.1881 | 0.2918 | 0.3105 |

## 6. Observations
- Logistic Regression achieved the highest accuracy (90.12%) and AUC (0.9056).
- SVM achieved the highest precision (0.6503), but its recall was low (0.1881).
- Gaussian Naive Bayes achieved the highest recall (0.5198).
- Decision Tree achieved the highest F1 Score (0.4701).
- The dataset has an imbalanced target distribution, so accuracy alone is insufficient for judging model performance.
- Logistic Regression demonstrated strong overall performance based on accuracy, AUC, and MCC.

## 7. Streamlit Application
The Streamlit application provides:
- CSV file upload
- Model selection
- Evaluation metrics display
- Confusion matrix visualization
- Classification report
- Comparison of all five trained models

## 8. Project Structure

```text
Bank_Marketing_ML/
├── data/
│   └── bank-full.csv
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── train.py
├── app.py
├── requirements.txt
├── model_results.csv
├── test_data.csv
└── README.md
```

## 9. How to Run the Project

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train models:

```bash
python train.py
```

Launch Streamlit application:

```bash
python -m streamlit run app.py
```

## 10. Conclusion
This project demonstrates the use of five machine learning classification algorithms for predicting bank marketing subscription outcomes. The models were evaluated using multiple performance metrics, and the results were presented through an interactive Streamlit application.

# 11. Project Links
- **GitHub Repository:** https://github.com/Bits560085/Bank_Marketing_ML
- **Live Streamlit Application:** https://bankmarketingml-xphzai3xxmx24xu8hve7mc.streamlit.app/
- **Dataset Source:** https://archive.ics.uci.edu/dataset/222/bank+marketing