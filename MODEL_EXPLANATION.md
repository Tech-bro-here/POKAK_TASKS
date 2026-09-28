Model Explanation (Task 3)

1. Why I chose this model
I compared three models: Decision Tree, Random Forest, and XGBoost

Decision Tree
Random Forest
XGBoost
5-fold CV accuracy

0.781
0.838
0.831
Test F1 (churn class)
0.513
0.578
0.596
I chose Random Forest because
it has the best performance on cross-validation
and
the dataset has both categorical and numeric features with non-linear relationships
which tree ensemble models can capture better than linear/logistic regression. The average over multiple trees also reduces overfitting, and the inherent feature importance helps interpret the model.
Though XGBoost has slightly higher test F1 (0.596 vs 0.578), the Random Forest has better CV accuracy (0.838 vs 0.831), so it has more consistent performance. The two models are quite similar in power, but I prefer Random Forest for its simplicity and easier hyperparameter tuning.
2. How the features affect the prediction
The random forest regressor's feature importances (see feature_importance.png):
Feature
TotalCharges
MonthlyCharges
Contract
tenure
OnlineSecurity
TechSupport
Importance
0.140
0.139
0.129
0.122
0.085
0.075

The most important factors for customer churn are tenure, contract type, and charges. Customers on month-to-month contracts with shorter tenures and higher monthly charges are more likely to churn. Having online security and tech support as add-ons also appear to reduce the chance of churn.
The relative importance of features tells us what variables to focus on when trying to reduce churn. It does not mean that those features cause churn - the relationship may be non-causal. For example, customers with higher income may be more likely to choose a month-to-month contract, but it's also possible that people with month-to-month contracts have more flexibility and are therefore more likely to churn when their needs change.

3. What else could be done to improve the model
Some ideas to improve the model:
Hyperparameter tuning with grid search or bayesian optimization with Optuna.
Fix the SMOTE leakage by using an imblearn pipeline that applies SMOTE only on the training folds during cross-validation.
Use one-hot encoding instead of label encoding for categorical variables.
Create new features like charges per month of tenure or count of add-on services.
Tune the decision threshold to favor recall over precision since false negatives (failing to predict a churner) might be more costly than false positives.
Use SHAP values to explain individual predictions.
Make the model production-ready by containerizing it with Docker, adding input validation (restricting to the same categories as in the training data), logging, monitoring, and scheduled retraining.