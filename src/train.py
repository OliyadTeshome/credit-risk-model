import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os


def main():
    # Load processed features
    data_path = os.path.join('data', 'processed', 'customer_risk_target.csv')
    df = pd.read_csv(data_path)
    features = ['recency_days', 'frequency', 'monetary_total', 'monetary_avg', 'cluster']
    target = 'is_high_risk'
    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define models and hyperparameters
    models = {
        'LogisticRegression': (
            LogisticRegression(max_iter=1000),
            {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs']}
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]}
        ),
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 10]}
        )
    }

    best_score = 0
    best_model = None
    best_model_name = None
    best_params = None
    best_metrics = None

    mlflow.set_experiment('credit-risk-model')

    for name, (model, params) in models.items():
        with mlflow.start_run(run_name=name):
            search = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1)
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            y_proba = search.predict_proba(X_test)[:, 1] if hasattr(search, 'predict_proba') else None

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
            }

            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(search.best_estimator_, name + "_model")

            print(f"\nModel: {name}")
            print(classification_report(y_test, y_pred))
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

            if metrics['roc_auc'] > best_score:
                best_score = metrics['roc_auc']
                best_model = search.best_estimator_
                best_model_name = name
                best_params = search.best_params_
                best_metrics = metrics

    # Register best model
    if best_model is not None:
        print(f"\nBest model: {best_model_name} (ROC-AUC: {best_score:.4f})")
        mlflow.sklearn.log_model(best_model, "best_model", registered_model_name="CreditRiskBestModel")
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, os.path.join('models', 'credit_risk_model.joblib'))
        print(f"Best model saved to models/credit_risk_model.joblib")

if __name__ == '__main__':
    main() 