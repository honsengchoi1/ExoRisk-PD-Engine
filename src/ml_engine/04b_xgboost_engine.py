"""
ExoRisk_PD_Engine: Phase 4b - XGBoost Training & Validation Hub
Author: Hon Seng Choi, Principal Quantitative Architect
Context: Ingests macro-sensitized design matrices, calculates algorithmic class weights, 
         trains XGBoost PD engine, and evaluates Out-of-Time (OOT) generalization.
Output: Serialized XGBoost Model (JSON) and Enterprise Risk Telemetry.
"""

import logging
import sys
from pathlib import Path
import time

import pandas as pd
import numpy as np
import xgboost as xgb
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
    
    models_dir.mkdir(parents=True, exist_ok=True)
    model_out_path = models_dir / "pd_engine_v1.json"

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
    # predict_proba returns a 2D array [Prob(0), Prob(1)]. We slice [:, 1] to get PD.
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

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

    # 7. Serialization
    logger.info(f"Serializing trained engine to {model_out_path}...")
    clf.save_model(model_out_path)

    exec_time = time.perf_counter() - start_time
    logger.info(f"Phase 4b Complete. Total execution time: {exec_time:.2f} seconds.")

if __name__ == "__main__":
    main()