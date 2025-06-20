import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

def train_and_evaluate(X, y, model, config, scaler):
    # Set experiment name in MLflow
    mlflow.set_experiment("FraudRadar")

    with mlflow.start_run():
        print("[INFO] Starting training...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=config['random_state'],
            stratify=y
        )

        print(f"[INFO] Before SMOTE: {y_train.value_counts().to_dict()}")
        # Apply SMOTE to balance training data
        smote = SMOTE(random_state=config['smote']['random_state'])
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        print(f"[INFO] After SMOTE: {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

        # Train model
        model.fit(X_train_sm, y_train_sm)
        print("[INFO] Model training completed.")

        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_proba)

        # Log hyperparameters
        mlflow.log_params(config['model'])

        # Log metrics
        mlflow.log_metric("precision", report['1']['precision'])
        mlflow.log_metric("recall", report['1']['recall'])
        mlflow.log_metric("f1_score", report['1']['f1-score'])
        mlflow.log_metric("roc_auc", auc)

        # Save model & scaler to disk
        os.makedirs(config['model_dir'], exist_ok=True)
        model_path = os.path.join(config['model_dir'], 'fraud_model.pkl')
        scaler_path = os.path.join(config['model_dir'], 'scaler.pkl')

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"[INFO] Model saved to {model_path}")
        print(f"[INFO] Scaler saved to {scaler_path}")

        # Log model artifact to MLflow
        mlflow.sklearn.log_model(model, artifact_path="fraud_rf_model")

        print("[INFO] Training run logged to MLflow.")
