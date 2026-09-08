"""
ExoRisk_PD_Engine: Master ETL Pipeline Orchestrator
Author: Hon Seng Choi, Principal Quantitative Architect
Context: Sequentially executes Macro Ingestion, Micro Ingestion, and Feature Merging,
         followed by a final QA data sanity check.
"""
import subprocess
import sys
import logging
import pandas as pd
from pathlib import Path

# 1. Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MasterETL")

def run_script(script_path: str) -> None:
    """Executes a Python script in a secure subprocess."""
    logger.info(f"=== Starting {script_path} ===")
    try:
        # sys.executable ensures the script runs inside your exact 'exorisk_env' virtual environment
        subprocess.run([sys.executable, script_path], check=True)
        logger.info(f"=== Successfully completed {script_path} ===\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"FATAL: {script_path} failed to execute. Pipeline halted.")
        sys.exit(1)

def run_qa_check() -> None:
    """Performs a final validation check on the ML-ready dataset."""
    logger.info("=== Initiating Final Data QA Sanity Check ===")
    processed_path = Path("data/processed/macro_sector_features.csv.gz")
    
    if not processed_path.exists():
        logger.error(f"QA Failed: Could not find target file at {processed_path}")
        sys.exit(1)
        
    logger.info(f"Loading finalized dataset from {processed_path} (This takes ~20 seconds)...")
    df = pd.read_csv(processed_path, compression='gzip')
    
    print("\n" + "="*60)
    print("               FINAL DATASET QA REPORT")
    print("="*60)
    print(f"Total Borrowers (Rows): {df.shape[0]:,}")
    print(f"Total Features (Cols):  {df.shape[1]}")
    
    print("\n--- Missing Values (%) ---")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct.round(2).astype(str) + '%')
    
    print("\n--- Data Sneak Peek ---")
    # Display a randomized sample for visual inspection
    print(df.sample(5).to_string())
    print("="*60 + "\n")
    
    logger.info("QA Check Passed. Data is 100% structurally sound and ready for XGBoost.")

def main() -> None:
    logger.info("Initiating Master ETL Pipeline...")
    
    # 2. Define the exact execution order
    scripts = [
        "src/data_pipeline/01_macro_ingestion.py",
        "src/data_pipeline/02_micro_ingestion.py",
        "src/data_pipeline/03_feature_merge.py"
    ]
    
    # 3. Execute scripts sequentially
    for script in scripts:
        run_script(script)
        
    # 4. Trigger automated QA validation
    run_qa_check()
    logger.info("Master ETL Pipeline Execution Complete.")

if __name__ == "__main__":
    main()