"""
ExoRisk_PD_Engine: Phase 2 Micro Ingestion
Author: Hon Seng Choi, Principal Quantitative Architect
Context: Idempotent extraction of the LendingClub dataset via Kaggle API.
Output: Raw CSV secured in data/raw/
"""

import os
import sys
import getpass
import logging
import subprocess
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
    
    # ---------------------------------------------------------
    # 1. Path Routing & Idempotent Gatekeeper
    # ---------------------------------------------------------
    if expected_file.exists():
        logger.info(f"Target data already exists at: {expected_file}")
        logger.info("Bypassing Kaggle extraction to conserve bandwidth and compute.")
        sys.exit(0)
        
    logger.info(f"Data not found. Initiating secure Kaggle API handshake for '{dataset_slug}'...")
# ---------------------------------------------------------
    # 2. Credential Management 
    # ---------------------------------------------------------
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_user or not kaggle_key:
        logger.warning("Kaggle credentials not found in environment variables.")
        print("\n--- Kaggle API Credentials Required ---")
        print("You can get these by generating a 'New API Token' in your Kaggle Account Settings.")
        
        # Enterprise Standard: Masking input via getpass
        raw_user = getpass.getpass("Please paste your KAGGLE_USERNAME and press Enter: ")
        raw_key = getpass.getpass("Please paste your KAGGLE_KEY and press Enter: ")
        
        # Aggressive sanitization to prevent hidden character bugs from pasting
        clean_user = raw_user.strip().replace('"', '').replace("'", "")
        clean_key = raw_key.strip().replace('"', '').replace("'", "")
        
        if not clean_user or not clean_key:
            logger.error("Missing username or key. Pipeline execution aborted.")
            sys.exit(1)
            
        os.environ["KAGGLE_USERNAME"] = clean_user
        os.environ["KAGGLE_KEY"] = clean_key
        print("Kaggle credentials accepted. Resuming extraction...\n")

    # ---------------------------------------------------------
    # 3. Native API Extraction via Subprocess
    # ---------------------------------------------------------
    try:
        logger.info("Initiating Kaggle CLI extraction via Python subprocess...")
        
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