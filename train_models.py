import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")
PLOTS_DIR = ARTIFACTS_DIR / "plots"
COST_FN = 10.0  # Cost of a False Negative (Missed Fraud)
COST_FP = 1.0   # Cost of a False Positive (False Alarm)

def setup_directories():
    os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    data_path = Path("creditcard.csv")
    if not data_path.exists():
        raise FileNotFoundError("creditcard.csv not found.")
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    return df

def preprocess_data(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    # Train/Val/Test split (60/20/20)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full
    )

    # Scale Amount based on X_train only to avoid leakage
    scaler = StandardScaler()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_val["Amount"] = scaler.transform(X_val[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])
    
    if "Time" in X_train.columns:
        X_train = X_train.drop(columns=["Time"])
        X_val = X_val.drop(columns=["Time"])
        X_test = X_test.drop(columns=["Time"])
    
    # Save preprocessor
    joblib.dump(scaler, ARTIFACTS_DIR / "preprocessor.pkl")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X_train.columns.tolist()

def calculate_cost(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (fn * COST_FN) + (fp * COST_FP)

def tune_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 100)
    costs = [calculate_cost(y_true, y_prob, t) for t in thresholds]
    best_threshold = thresholds[np.argmin(costs)]
    return best_threshold, min(costs)

def plot_pr_curve(y_true, y_prob, model_name):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / f"{model_name.lower().replace(' ', '_')}_pr_curve.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

def train_decision_tree(X_train, y_train, X_val, y_val):
    print("Training Decision Tree...")
    # SMOTE + Class Weight as per requirements
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    model = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced')
    model.fit(X_res, y_res)
    
    y_prob_val = model.predict_proba(X_val)[:, 1]
    best_threshold, _ = tune_threshold(y_val, y_prob_val)
    
    joblib.dump(model, ARTIFACTS_DIR / "decision_tree_model.pkl")
    return model, best_threshold

def train_xgboost(X_train, y_train, X_val, y_val):
    print("Training XGBoost...")
    # scale_pos_weight for cost-sensitive learning
    ratio = (y_train == 0).sum() / y_train.sum()
    model = XGBClassifier(
        random_state=RANDOM_STATE,
        scale_pos_weight=ratio,
        eval_metric="logloss",
        n_estimators=100
    )
    model.fit(X_train, y_train)
    
    y_prob_val = model.predict_proba(X_val)[:, 1]
    best_threshold, _ = tune_threshold(y_val, y_prob_val)
    
    joblib.dump(model, ARTIFACTS_DIR / "xgboost_model.pkl")
    return model, best_threshold

def train_neural_network(X_train, y_train, X_val, y_val):
    print("Training Neural Network...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Calculate class weights manually for Keras if not using SMOTE, 
    # but here we use SMOTE as requested for NN.
    model.fit(
        X_res, y_res,
        epochs=50,
        batch_size=2048,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=0
    )
    
    y_prob_val = model.predict(X_val).flatten()
    best_threshold, _ = tune_threshold(y_val, y_prob_val)
    
    model.save(ARTIFACTS_DIR / "mlp_model.keras")
    return model, best_threshold

def evaluate_model(model, X_test, y_test, threshold, model_name):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.predict(X_test).flatten()
    
    y_pred = (y_prob >= threshold).astype(int)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_prob)
    cost = calculate_cost(y_test, y_prob, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    plot_pr_curve(y_test, y_prob, model_name)
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": cost
    }

def main():
    setup_directories()
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = preprocess_data(df)
    
    results = []
    
    # Decision Tree
    dt_model, dt_threshold = train_decision_tree(X_train, y_train, X_val, y_val)
    results.append(evaluate_model(dt_model, X_test, y_test, dt_threshold, "DecisionTree_SMOTE_ClassWeight"))
    
    # XGBoost
    xgb_model, xgb_threshold = train_xgboost(X_train, y_train, X_val, y_val)
    results.append(evaluate_model(xgb_model, X_test, y_test, xgb_threshold, "XGBoost_ScalePosWeight"))
    
    # Neural Network
    nn_model, nn_threshold = train_neural_network(X_train, y_train, X_val, y_val)
    results.append(evaluate_model(nn_model, X_test, y_test, nn_threshold, "TensorFlowMLP_SMOTE_ClassWeight"))
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(ARTIFACTS_DIR / "metrics_comparison.csv", index=False)
    
    # Identify best model based on cost (minimizing false negatives)
    best_model_info = results_df.loc[results_df['total_cost'].idxmin()]
    
    metadata = {
        "random_state": RANDOM_STATE,
        "cost_fn": COST_FN,
        "cost_fp": COST_FP,
        "target_column": "Class",
        "feature_columns": feature_cols,
        "best_model": best_model_info['model'],
        "thresholds": {
            "decision_tree": dt_threshold,
            "xgboost": xgb_threshold,
            "tensorflow_mlp": nn_threshold
        }
    }
    
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training complete. Best model: {best_model_info['model']} with cost {best_model_info['total_cost']}")

if __name__ == "__main__":
    main()
