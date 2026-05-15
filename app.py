import os
import json
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Configuration
DATA_PATH = Path(__file__).resolve().parent / "creditcard.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
RANDOM_STATE = 42

MODEL_OPTIONS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"]
IMBALANCE_OPTIONS = ["Undersampling", "SMOTE", "SMOTETomek", "None"]

# Constants for cost-benefit analysis
DEFAULT_COST_FN = 10.0
DEFAULT_COST_FP = 1.0

@st.cache_data
def load_and_preprocess_data() -> tuple[pd.DataFrame, list[str]]:
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}. Please ensure creditcard.csv is in the project root.")
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    feature_cols = [c for c in df.columns if c != "Class"]
    return df, feature_cols

def load_comparison_metrics():
    metrics_path = ARTIFACTS_DIR / "metrics_comparison.csv"
    if metrics_path.exists():
        return pd.read_csv(metrics_path)
    return None

def apply_imbalance_handling(X_train: pd.DataFrame, y_train: pd.Series, method: str):
    if method == "Undersampling":
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        return rus.fit_resample(X_train, y_train)
    if method == "SMOTE":
        sm = SMOTE(random_state=RANDOM_STATE)
        return sm.fit_resample(X_train, y_train)
    if method == "SMOTETomek":
        smt = SMOTETomek(random_state=RANDOM_STATE)
        return smt.fit_resample(X_train, y_train)
    return X_train, y_train

def build_model(model_name: str, y_train: pd.Series, input_dim: int):
    if model_name == "Logistic Regression":
        return LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced')
    if model_name == "Random Forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100, n_jobs=-1, class_weight='balanced')
    if model_name == "XGBoost":
        ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        return XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=100,
            scale_pos_weight=ratio,
            eval_metric="logloss",
            n_jobs=-1,
        )
    if model_name == "Neural Network":
        model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    return None

@st.cache_resource
def joblib.load(model_name: str, imbalance_method: str):
    df, feature_cols = load_and_preprocess_data()
    X = df[feature_cols]
    y = df["Class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale Amount based on X_train to avoid leakage
    scaler = StandardScaler()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])
    
    X_train_bal, y_train_bal = apply_imbalance_handling(X_train, y_train, imbalance_method)
    
    model = build_model(model_name, y_train_bal, X_train.shape[1])
    
    if model_name == "Neural Network":
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(
            X_train_bal, y_train_bal,
            epochs=20,
            batch_size=2048,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        y_proba = model.predict(X_test).flatten()
    else:
        model.fit(X_train_bal, y_train_bal)
        y_proba = model.predict_proba(X_test)[:, 1]

    return model, feature_cols, y_test, y_proba, scaler

def calculate_cost(y_true, y_proba, threshold, cost_fn, cost_fp):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (fn * cost_fn) + (fp * cost_fp), tn, fp, fn, tp

def validate_input(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    extra = [c for c in df.columns if c not in feature_cols]
    if extra:
        df = df.drop(columns=extra)

    df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if df.isna().any().any():
        df = df.fillna(df.median(numeric_only=True))
    return df

def main():
    st.set_page_config(page_title="Fraud Detection System", layout="wide")
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("""
    Develop a binary classification system to identify fraudulent credit card transactions. 
    This system addresses severe class imbalance using **SMOTE** and **Cost-Sensitive Learning**, 
    comparing **Decision Trees**, **XGBoost**, and **Neural Networks**.
    """)

    # Sidebar for parameters
    st.sidebar.header("⚙️ Model Configuration")
    model_name = st.sidebar.selectbox("Model", MODEL_OPTIONS, index=3)
    imbalance_method = st.sidebar.selectbox("Imbalance Handling", IMBALANCE_OPTIONS, index=1)
    
    st.sidebar.header("💰 Cost-Benefit Parameters")
    st.sidebar.info("FN Cost: Penalty for missing a fraud. FP Cost: Penalty for false alarm.")
    cost_fn = st.sidebar.number_input("Cost of False Negative", value=DEFAULT_COST_FN, step=1.0)
    cost_fp = st.sidebar.number_input("Cost of False Positive", value=DEFAULT_COST_FP, step=0.1)
    
    threshold = st.sidebar.slider("Decision Threshold", 0.01, 0.99, 0.50, 0.01)

    # Main application logic
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🎯 Single Prediction", "📂 Batch Prediction", "🏆 Model Comparison"])

    with st.spinner("Preparing model..."):
        model, feature_cols, y_test, y_proba, scaler = train_pipeline(model_name, imbalance_method)

    total_cost, tn, fp, fn, tp = calculate_cost(y_test, y_proba, threshold, cost_fn, cost_fp)
    
    with tab1:
        st.subheader("Performance Summary")
        c1, c2, c3, c4 = st.columns(4)
        
        y_pred = (y_proba >= threshold).astype(int)
        c1.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        c2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.4f}")
        c3.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
        c4.metric("Total Cost", f"${total_cost:,.2f}", delta=f"FN: {fn}, FP: {fp}", delta_color="inverse")

        st.write(f"**Confusion Matrix:** TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label=f'PR Curve (AP={ap:.4f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curve: {model_name}')
            ax.legend()
            st.pyplot(fig)
        
        with col_plot2:
            # Cost vs Threshold
            thresholds = np.linspace(0.01, 0.99, 50)
            costs = [calculate_cost(y_test, y_proba, t, cost_fn, cost_fp)[0] for t in thresholds]
            fig2, ax2 = plt.subplots()
            ax2.plot(thresholds, costs)
            ax2.axvline(threshold, color='red', linestyle='--', label=f'Current Threshold: {threshold}')
            ax2.set_xlabel('Threshold')
            ax2.set_ylabel('Total Cost')
            ax2.set_title('Cost vs. Decision Threshold')
            ax2.legend()
            st.pyplot(fig2)

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Importance")
            fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            st.bar_chart(fi.set_index("feature").head(15))

    with tab2:
        st.subheader("Manual Transaction Entry")
        cols = st.columns(4)
        input_values = {}
        for i, feat in enumerate(feature_cols):
            with cols[i % 4]:
                input_values[feat] = st.number_input(feat, value=0.0, format="%.6f", key=f"single_{feat}")
        
        if st.button("Predict Fraud"):
            input_df = pd.DataFrame([input_values])
            # Apply scaling to Amount
            input_df["Amount"] = scaler.transform(input_df[["Amount"]])
            
            if model_name == "Neural Network":
                prob = model.predict(input_df)[0][0]
            else:
                prob = model.predict_proba(input_df)[0][1]
            
            is_fraud = prob >= threshold
            st.subheader(f"Result: {'🚨 FRAUD' if is_fraud else '✅ NOT FRAUD'}")
            st.metric("Fraud Probability", f"{prob:.4f}")
            st.progress(float(prob))
            st.write(f"Confidence: {prob if is_fraud else 1-prob:.2%}")

    with tab3:
        st.subheader("Batch Prediction via CSV")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            input_df = pd.read_csv(uploaded_file)
            try:
                valid_df = validate_input(input_df, feature_cols)
                # Apply scaling to Amount
                valid_df["Amount"] = scaler.transform(valid_df[["Amount"]])
                
                if model_name == "Neural Network":
                    probs = model.predict(valid_df).flatten()
                else:
                    probs = model.predict_proba(valid_df)[:, 1]
                
                preds = np.where(probs >= threshold, "Fraud", "Not Fraud")
                results_df = pd.DataFrame({
                    "Fraud_Probability": probs,
                    "Prediction": preds,
                    "Confidence": np.where(preds == "Fraud", probs, 1 - probs)
                })
                
                st.write(results_df.head(10))
                st.download_button("Download Predictions", results_df.to_csv(index=False), "predictions.csv", "text/csv")
                
                st.subheader("Batch Distribution")
                st.bar_chart(results_df["Prediction"].value_counts())
            except Exception as e:
                st.error(f"Error processing file: {e}")

    with tab4:
        st.subheader("Model Comparison (Pre-calculated)")
        st.markdown("""
        These metrics are generated from the `train_models.py` pipeline, which performs a full evaluation 
        of the three core models requested in the requirements.
        """)
        comparison_df = load_comparison_metrics()
        if comparison_df is not None:
            st.dataframe(comparison_df.style.highlight_min(subset=['total_cost'], color='lightgreen'))
            
            # Visualization of costs
            fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
            comparison_df.plot(kind='bar', x='model', y='total_cost', ax=ax_comp, color='skyblue')
            ax_comp.set_title("Total Cost Comparison (Lower is Better)")
            ax_comp.set_ylabel("Cost")
            plt.xticks(rotation=45)
            st.pyplot(fig_comp)
        else:
            st.warning("No pre-calculated metrics found. Run `train_models.py` to generate them.")

if __name__ == "__main__":
    main()
