# Customer Churn Prediction

A small AI to predict that telecom customer will churn or not with a REST API to make predictions.
Below are the links to the documents describing each of the tasks.

Doc Purpose
[SETUP.md](SETUP.md) How to install and run the code
[MODEL_EXPLANATION.md](MODEL_EXPLANATION.md) Why we picked the model, how features impact the outcome, and what can be improved

## Task 1: Data Understanding & Preprocessing
Where: [costumer_churning.py](costumer_churning.py) section `TASK 1: Data understanding & preprocessing`
Dataset: [WA_Fn-UseC_-Telco-Customer-Churn.csv](WA_Fn-UseC_-Telco-Customer-Churn.csv) 7043 rows, mixed types

What is done in this section:
- Drop the customerID column
- Fill nan in TotalCharges (which were empty strings) with 0.0 and convert to float
- Encode categories with LabelEncoder (save the encoder to [encoders.pkl](encoders.pkl)), encode the target column Churn to 1/0
- Scale the data? No, since all the models are tree-based (see the code comments)
- Handle class imbalance with SMOTE (only on the train set)

The output are the [correlation heatmap](correlation_heatmap.png) and [encoders.pkl](encoders.pkl)

## Task 2: Model Building (Option A: Machine Learning)
Where: [costumer_churning.py](costumer_churning.py) section `TASK 2: Model building`

We trained three classifiers and compared their performance on the hold-out 20% test set. The metrics (precision, recall, and F1 are for the churn class):

Model Accuracy Precision Recall F1
Decision Tree 0.721 0.477 0.555 0.513
Random Forest 0.777 0.580 0.576 0.578
XGBoost 0.782 0.585 0.606 0.596

We used a random forest with the best validation performance and saved it to [customer_churn_model.pkl](customer_churn_model.pkl) (see Task 3 for details)

The output are [model_comparison.csv](model_comparison.csv) and [customer_churn_model.pkl](customer_churn_model.pkl)

## Task 3: AI Logic & Explanation
Where: [MODEL_EXPLANATION.md](MODEL_EXPLANATION.md)
Code: [costumer_churning.py](costumer_churning.py) section `TASK 3: Feature impact` (it produces the [feature_importance.png](feature_importance.png))

This document describes:
- Why we picked the model
- How features impact the prediction
- And what can be improved

## Task 4: Deployment / API
Where: [app.py](app.py)
The API has two endpoints:

- GET / - health check
- POST /predict - takes the customer data as JSON, outputs the prediction as JSON
Sample output:

{"prediction": "No Churn", "churn_probability": 0.17}

Invalid category values return HTTP 400
Start the server and try it in your browser at http://127.0.0.1:8000/docs

More details are in the [SETUP.md](SETUP.md)
## Task 5: Git & Documentation
- Code is committed to GitHub
- The [README.md](README.md) is this document
- [SETUP.md](SETUP.md) describes how to run the code
- [MODEL_EXPLANATION.md](MODEL_EXPLANATION.md) describes the model choice
The repo structure is as follows:
```
├── costumer_churning.py         # Tasks 1, 2, 3 (code)
├── app.py                # Task 4 (API)
├── requirements.txt           # dependencies
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # dataset
├── customer_churn_model.pkl       # trained model
├── encoders.pkl             # saved label encoders
├── model_comparison.csv         # Task 2 results
├── feature_importance.png        # Task 3 chart
├── correlation_heatmap.png       # Task 1 chart
├── README.md
├── SETUP.md
└── MODEL_EXPLANATION.md
```