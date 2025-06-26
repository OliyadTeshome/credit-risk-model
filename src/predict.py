import joblib
import numpy as np
import os

def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join('models', 'credit_risk_model.joblib')
    return joblib.load(model_path)

def predict_risk(recency_days, frequency, monetary, model=None):
    if model is None:
        model = load_model()
    X = np.array([[recency_days, frequency, monetary]])
    proba = model.predict_proba(X)[0, 1]
    return proba

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--recency_days', type=float, required=True)
    parser.add_argument('--frequency', type=float, required=True)
    parser.add_argument('--monetary', type=float, required=True)
    args = parser.parse_args()
    model = load_model()
    risk = predict_risk(args.recency_days, args.frequency, args.monetary, model)
    print(f'Predicted risk score: {risk:.2f}') 