"""
ExoRisk_PD_Engine: Phase 2 Micro Ingestion
Author: Hon Seng Choi, Principal Quantitative Architect
Context: Idempotent extraction of the LendingClub dataset via Kaggle API.
Output: Raw CSV secured in data/raw/
"""

import os
import sys
import logging
from pathlib import Path

# Configure Executive Telemetry Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MicroIngestion")

def get_project_root() -> Path:
    """Dynamically resolves the project root based on this script's location."""
    return Path(__file__).resolve().parents[2]

def main() -> None:
    logger.info("Initiating Phase 2: Micro Ingestion (Idempotent Extractor)")
    
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # The specific dataset and expected output file
    dataset_slug = "wordsforthewise/lending-club"
    expected_file = raw_dir / "accepted_2007_to_2018Q4.csv.gz"
    
    # 1. Idempotent Existence Check
    if expected_file.exists():
        logger.info(f"Target data already exists at: {expected_file}")
        logger.info("Bypassing Kaggle extraction to conserve bandwidth and compute.")
        sys.exit(0)
        
    logger.info(f"Data not found. Initiating secure Kaggle API handshake for '{dataset_slug}'...")
    
# 2. Native API Extraction via Subprocess
    try:
        import subprocess
        
        logger.info("Initiating Kaggle CLI extraction via Python subprocess...")
        
        # Execute the robust CLI command directly from within Python
        subprocess.run([
            "kaggle", "datasets", "download", 
            "-d", dataset_slug, 
            "-p", str(raw_dir), 
            "--unzip"
        ], check=True)
        
        # Verification
        if expected_file.exists():
            file_size_mb = expected_file.stat().st_size / (1024 * 1024)
            logger.info(f"Extraction Complete. File secured: {file_size_mb:.2f} MB")
        else:
            logger.error("Download completed, but expected CSV was not found.")
            sys.exit(1)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"FATAL: Kaggle CLI extraction failed. Ensure Kaggle is installed and authenticated. Details: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL: Unexpected error during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()