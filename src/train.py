import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def main():
    # Load processed RFM features
    rfm_path = os.path.join('data', 'processed', 'rfm.csv')
    df = pd.read_csv(rfm_path)
    # Assume 'default' column is the target (1 = default, 0 = no default)
    X = df[['recency_days', 'frequency', 'monetary']]
    y = df['default']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf, os.path.join('models', 'credit_risk_model.joblib'))

if __name__ == '__main__':
    main() 