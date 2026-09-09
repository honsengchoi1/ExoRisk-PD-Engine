"""
ExoRisk_PD_Engine: Phase 3 Feature Assembly & Macro Mapping
Author: Hon Seng Choi, Principal Quantitative Architect
Context: Executes vectorized Regex mapping of job titles into 14 macro sectors.
Refactor: Switched to Parquet for dimensional lookup storage, added 'id' for key-joins.
Output: Snappy-compressed Parquet Lookup Table (ID, Sector).
"""

import pandas as pd
import logging
import sys
from pathlib import Path

# 1. Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeatureMerge")

def main():
    logger.info("Initiating Phase 3: Feature Merging & Macro Mapping (Refactored)")
    
    # 2. Path Routing
    # Adjust this path if you are using dynamic resolution (__file__) in your actual environment
    raw_path = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Saving clean dimensional data as lightweight Parquet for fast lookups
    output_path = processed_dir / "macro_sector_lookup.parquet"

    # 3. Idempotent Gatekeeper
    if output_path.exists():
        logger.info(f"Dimensional data already exists at {output_path}. Bypassing Phase 3.")
        sys.exit(0)

    if not raw_path.exists():
        logger.error("Raw data missing. Please run 02_micro_ingestion.py first.")
        sys.exit(1)

    # 4. Data Extraction (Refactored)
    logger.info("Loading necessary text and key columns from raw payload...")
    
    # Only load the 'id' (for merging) and the 'emp_title' (for mapping)
    cols_to_load = ['id', 'emp_title']
    df_map = pd.read_csv(raw_path, usecols=cols_to_load, low_memory=False, compression='gzip')

    # Clean the emp_title column before mapping
    emp_titles_clean = df_map['emp_title'].dropna().str.lower().str.strip()

    # 5. The Enterprise Regex Dictionary
    logger.info("Applying Regex Macro Sector mappings (This may take ~30 seconds)...")
    regex_rules = {
        'Management': r'\b(?:manager|supervisor|director|president|vp|owner|executive|chief|ceo|cfo|coo)\b',
        'Healthcare': r'\b(?:nurse|rn|medical|doctor|dentist|therapist|pharmacy|clinical|hospital|paramedic|lpn)\b',
        'Education': r'\b(?:teacher|professor|instructor|school|college|university|student|coach|principal)\b',
        'Technology': r'\b(?:it|software|developer|programmer|tech|technician|network|data|systems)\b',
        'Finance': r'\b(?:finance|accounting|accountant|bank|banker|cpa|financial|auditor|payroll)\b',
        'Construction_Trades': r'\b(?:construction|carpenter|plumber|electrician|mechanic|welder|contractor|hvac)\b',
        'Government_Military': r'\b(?:government|military|police|army|navy|usaf|usmc|state|city|federal|usps|officer)\b',
        'Sales_Retail': r'\b(?:sales|retail|associate|cashier|store|buyer|merchandiser|salesperson)\b',
        'Logistics_Transport': r'\b(?:driver|truck|warehouse|delivery|logistics|shipping|postal|freight)\b',
        'Legal': r'\b(?:attorney|lawyer|legal|paralegal|judge)\b',
        'Hospitality_Food': r'\b(?:chef|cook|restaurant|bartender|server|waitress|waiter|hotel|food)\b',
        'Real_Estate': r'\b(?:real estate|realtor|property|leasing|broker)\b',
        'Engineering_Science': r'\b(?:engineer|engineering|architect|scientist|chemist)\b',
        'Administrative': r'\b(?:admin|assistant|clerk|receptionist|secretary|coordinator|office)\b'
    }

    # Initialize new column
    df_map['macro_sector'] = 'Other'

    # Apply mapping
    for sector, pattern in regex_rules.items():
        mask = emp_titles_clean.str.contains(pattern, na=False, regex=True)
        # Map the matches back to the dataframe
        df_map.loc[mask.reindex(df_map.index, fill_value=False), 'macro_sector'] = sector

    # Drop raw text, keep only ID and Sector mapping
    df_map = df_map[['id', 'macro_sector']]

    # 6. Load to Processed Storage (Refactored to Parquet)
    logger.info(f"Mapping complete. Saving clean lookup table to {output_path}...")
    
    # Saving as snappy-compressed Parquet for storage optimization and fast I/O
    df_map.to_parquet(output_path, compression='snappy')
    logger.info("Phase 3 Refactor Complete.")

if __name__ == "__main__":
    main()