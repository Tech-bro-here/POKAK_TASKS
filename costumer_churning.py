"""Customer Churn Prediction - Tasks 1, 2 and 3 (preprocessing, modelling, evaluation)."""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# TASK 1: Data understanding & preprocessing
# ---------------------------------------------------------------------------
df = pd.read_csv("./WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.info()
print(df.isnull().sum())

# customerID is an identifier, not a feature
df = df.drop(columns=["customerID"])

# TotalCharges is stored as text; blank strings are hidden missing values
print("Blank TotalCharges rows:", (df["TotalCharges"] == " ").sum())
df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"}).astype(float)

print(df["Churn"].value_counts())

# Quick EDA plots
plt.figure(figsize=(8, 4))
sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Encode categorical features and save encoders for the API
encoders = {}
for column in df.select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Feature scaling skipped: Decision Tree, Random Forest and XGBoost are
# tree-based and invariant to feature scale.

# Train / test split
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Handle class imbalance on the training set only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(y_train_smote.value_counts())

# ---------------------------------------------------------------------------
# TASK 2: Model building (Option A - Machine Learning)
# ---------------------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42),
}

# 5-fold cross-validation on the training data
for name, model in models.items():
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring="accuracy")
    print(f"{name} CV accuracy: {np.mean(scores):.3f}")

# Fit every model and evaluate on the untouched test set
rows = []
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
    })

results = pd.DataFrame(rows).round(3)
print(results)
results.to_csv("model_comparison.csv", index=False)

# Selected model: Random Forest
rfc = models["Random Forest"]
y_pred = rfc.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

with open("customer_churn_model.pkl", "wb") as f:
    pickle.dump({"model": rfc, "features_names": X.columns.tolist()}, f)

# ---------------------------------------------------------------------------
# TASK 3: Feature impact (supports the explanation in README.md)
# ---------------------------------------------------------------------------
importances = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(10))

plt.figure(figsize=(7, 4))
importances.head(10).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# ---------------------------------------------------------------------------
# Sanity check: load saved artifacts and predict one sample
# ---------------------------------------------------------------------------
sample = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}
sample_df = pd.DataFrame([sample])
for col, enc in encoders.items():
    if col in sample_df.columns:
        sample_df[col] = enc.transform(sample_df[col])
print("Prediction:", "Churn" if rfc.predict(sample_df[X.columns])[0] == 1 else "No Churn")
print("Probabilities:", rfc.predict_proba(sample_df[X.columns]))
