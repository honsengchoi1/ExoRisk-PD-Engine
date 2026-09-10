"""
ExoRisk_PD_Engine: Phase 4b - XGBoost Training & Validation Hub
Author: Hon Seng Choi, Principal Quantitative Architect
Context: Ingests macro-sensitized design matrices, calculates algorithmic class weights, 
         trains XGBoost PD engine, and evaluates Out-of-Time (OOT) generalization.
Output: Serialized XGBoost Model (JSON), Enterprise Risk Telemetry, and MRM Exhibits.
"""

import logging
import sys
import time
import json
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Configure Executive Telemetry Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("XGBoost_Engine")

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def main() -> None:
    start_time = time.perf_counter()
    logger.info("Initiating Phase 4b: XGBoost PD Engine Training & Validation")

    # 1. Dynamic Path Routing
    project_root = get_project_root()
    data_dir = project_root / "data" / "processed" / "model_matrices"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    
    # Ensure directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Ingesting Design Matrices
    logger.info("Loading Parquet design matrices into memory...")
    try:
        X_train = pd.read_parquet(data_dir / "X_train.parquet")
        X_test = pd.read_parquet(data_dir / "X_test.parquet")
        y_train = pd.read_parquet(data_dir / "y_train.parquet")['target_default']
        y_test = pd.read_parquet(data_dir / "y_test.parquet")['target_default']
    except FileNotFoundError:
        logger.error("Matrices not found. Please run 04a_model_data_prep.py first.")
        sys.exit(1)

    # 3. Algorithmic Balancing (scale_pos_weight)
    logger.info("Calculating native cost-sensitive weight ratio...")
    num_negatives = (y_train == 0).sum()
    num_positives = (y_train == 1).sum()
    pos_weight = num_negatives / num_positives
    
    logger.info(f"Class Imbalance Detected. Negative/Positive Ratio: {pos_weight:.2f}")

    # 4. Engine Architecture Configuration
    logger.info("Initializing XGBoost hyperparameters...")
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=pos_weight,
        eval_metric='auc',
        max_depth=5,              # Constrained depth to prevent overfitting macro noise
        learning_rate=0.05,       # Slower learning rate for better probability calibration
        n_estimators=300,         # Number of boosting rounds
        subsample=0.8,            # Stochastic sampling to reduce variance
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1                 # Utilize all CPU cores
    )

    # 5. Training execution
    logger.info(f"Commencing Engine Training on {len(X_train):,} records (This may take a few minutes)...")
    clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )

    # 6. Out-Of-Time (OOT) Validation
    logger.info("Generating Probability of Default (PD) predictions for OOT cohort...")
    
    # Raw probability under cost-sensitive weighting
    y_pred_raw = clf.predict_proba(X_test)[:, 1]

    # Analytic Bayesian calibration: correct log-odds bias introduced by scale_pos_weight
    logger.info("Applying closed-form Bayesian probability calibration...")
    y_pred_proba = y_pred_raw / (y_pred_raw + pos_weight * (1.0 - y_pred_raw))

    # Calculate Enterprise Risk Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)

    print("\n" + "="*60)
    print("      EXORISK OOT VALIDATION METRICS (2017-2018 COHORT)")
    print("="*60)
    print(f"ROC-AUC (Global Ranking):      {roc_auc:.4f}")
    print(f"PR-AUC  (Minority Precision):  {pr_auc:.4f}")
    print(f"Brier Score (Calibration):     {brier:.4f}")
    print("="*60 + "\n")

   # --- 7. Model Risk Governance Exhibits & Serialization ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    import json

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # A. Calibration Distribution Exhibit
    logger.info("Exporting Calibration Distribution Exhibit to reports/...")
    plt.figure(figsize=(10, 5))
    sns.histplot(y_pred_raw, color='red', label='Raw PD (Inflated by pos_weight)', kde=True, stat='density', alpha=0.5)
    sns.histplot(y_pred_proba, color='blue', label='Calibrated PD (Bayesian)', kde=True, stat='density', alpha=0.5)
    plt.title('ExoRisk: Pre- vs. Post-Calibration Probability Distribution', fontweight='bold')
    plt.xlabel('Probability of Default (PD)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(reports_dir / "calibration_plot.png", dpi=300)
    plt.close()

    # C. Engine & Metadata Serialization
    model_path = models_dir / "pd_engine_v1.json"
    logger.info(f"Serializing trained engine to {model_path}...")
    clf.save_model(model_path)
    
    meta_path = models_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"scale_pos_weight": float(pos_weight)}, f)
    logger.info(f"Serialized dynamic model metadata to {meta_path}...")

    end_time = time.perf_counter()
    logger.info(f"Phase 4b Complete. Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()