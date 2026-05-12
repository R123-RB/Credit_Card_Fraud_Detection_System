# Credit Card Fraud Detection System

## Project Overview
This project builds a binary fraud detection system on the Kaggle credit card transaction dataset with a production-oriented objective: reduce **false negatives** (missed fraud) while keeping false positives under control.

The pipeline compares three models:
- Decision Tree (`SMOTE + class_weight`)
- XGBoost (`scale_pos_weight`, no SMOTE)
- TensorFlow MLP (`SMOTE + class_weight`)

Model selection is not based on accuracy. It is based on:
- Precision
- Recall
- F1-score
- PR-AUC
- Confusion Matrix
- Cost-based objective:
  - `Total Cost = FN * Cost_FN + FP * Cost_FP`

---

## Why This Pipeline Is Leakage-Safe
- Train/test split is done before any resampling.
- Validation split is created from training data only.
- `StandardScaler` is fit on training data only.
- SMOTE is applied only on the training split.
- Decision threshold is tuned on validation data and evaluated once on test data.

---

## Repository Files
- `train_models.py` - full training, threshold tuning, evaluation, and artifact export
- `app.py` - Streamlit deployment app (manual input + CSV upload)
- `requirements.txt` - dependencies
- `creditcard.csv` - dataset input
- `artifacts/` - created after training:
  - `metrics_comparison.csv`
  - model files (`decision_tree_model.pkl`, `xgboost_model.pkl`, `mlp_model.keras`, `best_model.pkl/.keras`)
  - `preprocessor.pkl`
  - `metadata.json`
  - PR curves and confusion matrices in `artifacts/plots/`

---

## How to Run Locally
1. Create and activate environment:
   - Windows (PowerShell):
     - `python -m venv tfenv`
     - `.\tfenv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Train and generate artifacts:
   - `python train_models.py`
4. Launch app:
   - `streamlit run app.py`

---

## Streamlit Features
- Manual transaction input fields
- CSV batch upload
- Probability score per transaction
- Threshold-aware prediction (`Fraud` / `Not Fraud`)
- Confidence visualization
- Feature importance table/chart for Decision Tree and XGBoost
- Missing value and format handling with clear error messages
- Download predictions as CSV

---

## Deployment Steps
### Streamlit Community Cloud
1. Push project to GitHub.
2. Ensure `creditcard.csv` is not too large for deployment constraints; consider loading from cloud storage in production.
3. Set entry point to `app.py`.
4. Add `requirements.txt`.
5. Deploy and verify artifact path behavior.

### Production Suggestion
Run `train_models.py` in CI/CD or scheduled retraining job, version artifacts, and deploy model + metadata atomically.

---

## Recommended Next Improvements
- Add cross-validated threshold tuning to reduce variance.
- Add probability calibration (Platt scaling / isotonic) before threshold optimization.
- Add drift monitoring (feature drift + fraud-rate drift).
- Add temporal validation split to simulate real fraud detection chronology.
- Add explainability for individual predictions (e.g., SHAP for XGBoost).
- Add API service layer (FastAPI) for low-latency scoring.
