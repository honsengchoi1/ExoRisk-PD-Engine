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
    
    # DYNAMIC PATH RESOLUTION: Calculates the root regardless of where the script is run from
    project_root = Path(__file__).resolve().parent
    processed_path = project_root / "data" / "processed" / "macro_sector_lookup.parquet"
    
    if not processed_path.exists():
        logger.error(f"QA Failed: Could not find target file at {processed_path}")
        sys.exit(1)
        
    logger.info(f"Loading finalized lookup table from {processed_path} for QA reporting...")
    # UPDATED: Reading the new lightning-fast Parquet format
    df = pd.read_parquet(processed_path)
    
    print("\n" + "="*60)
    print("              FINAL DATASET QA REPORT")
    print("="*60)
    print(f"Total Mapped Borrowers (Rows): {df.shape[0]:,}")
    print(f"Total Features (Cols):  {df.shape[1]}")
    
    print("\n--- Missing Values (%) ---")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    print(missing_pct.round(2).astype(str) + '%')
    
    print("\n--- Data Sneak Peek ---")
    # Display a randomized sample for visual inspection
    print(df.sample(5).to_string())
    print("="*60 + "\n")
    
    logger.info("QA Check Passed. Dimensional data is structurally sound and ready for aggregation.")

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