"""
ExoRisk_PD_Engine: Phase 5 - SHAP Model Explainability
Author: Hon Seng Choi, Principal Quantitative Architect
Context: Deploys Shapley Additive exPlanations (SHAP) to open the XGBoost black box.
         Uses an elite Python namespace-injection patch to resolve the XGBoost 2.x UBJ bug.
Output: High-resolution SHAP Summary Plot (PNG).
"""

import logging
import sys
import builtins
from pathlib import Path

import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Configure Executive Telemetry Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SHAP_Explainer")

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def main() -> None:
    logger.info("Initiating Phase 5: SHAP Matrix Generation & Visualization")

    # 1. Dynamic Path Routing
    project_root = get_project_root()
    data_dir = project_root / "data" / "processed" / "model_matrices"
    models_dir = project_root / "models"
    report_dir = project_root / "reports"
    
    report_dir.mkdir(exist_ok=True)
    model_path = models_dir / "pd_engine_v1.json"
    out_path = report_dir / "shap_summary_plot.png"

    # 2. Ingestion
    logger.info("Loading OOT Test Matrix and serialized Engine...")
    try:
        X_test = pd.read_parquet(data_dir / "X_test.parquet")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    except FileNotFoundError:
        logger.error("Data or model missing. Please ensure Phase 4b is complete.")
        sys.exit(1)

    # Sample 10,000 records for compute efficiency
    logger.info("Sampling 10,000 OOT records for Shapley value computation...")
    X_sample = X_test.sample(n=10000, random_state=42)

    # === ELITE MLOPS NAMESPACE INJECTION PATCH ===
    # SHAP will try to call float("[5E-1]") and crash. 
    # We surgically inject a safe_float wrapper directly into SHAP's internal module namespace.
    logger.info("Injecting Python Namespace Patch into SHAP internal modules...")
    
    def safe_float(x):
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            x = x.strip('[]') # Strips the brackets so 'float' can read the math
        return builtins.float(x)
        
    shap.explainers._tree.float = safe_float
    # ==============================================

    # 3. SHAP Computation
    logger.info("Deploying SHAP TreeExplainer (This may take 30-60 seconds)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 4. Rendering the Executive Visual
    logger.info("Rendering SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values, 
        X_sample, 
        max_display=20,
        show=False
    )
    
    # 5. Serialization
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Phase 5 Complete. Executive visual saved to: {out_path}")

if __name__ == "__main__":
    main()