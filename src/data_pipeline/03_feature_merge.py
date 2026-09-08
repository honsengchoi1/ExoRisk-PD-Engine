import pandas as pd
import logging
from pathlib import Path
import sys

# 1. Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FeatureMerge")

def main():
    logger.info("Initiating Phase 3: Feature Merging & Macro Mapping")
    
    # 2. Path Routing
    raw_path = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # We will save the clean data as a compressed GZIP to keep it lightweight
    output_path = processed_dir / "macro_sector_features.csv.gz"

    # 3. Idempotent Gatekeeper
    if output_path.exists():
        logger.info(f"Clean data already exists at {output_path}. Bypassing Phase 3.")
        sys.exit(0)

    if not raw_path.exists():
        logger.error("Raw data missing. Please run 02_micro_ingestion.py first.")
        sys.exit(1)

    # 4. Data Extraction
    logger.info("Loading essential columns from raw payload...")
    cols_to_load = ['loan_amnt', 'term', 'int_rate', 'emp_title', 'loan_status']
    df = pd.read_csv(raw_path, usecols=cols_to_load, low_memory=False, compression='gzip')

    # Clean the emp_title column before mapping
    emp_titles_clean = df['emp_title'].dropna().str.lower().str.strip()

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
    df['macro_sector'] = 'Other'

    # Apply mapping
    for sector, pattern in regex_rules.items():
        mask = emp_titles_clean.str.contains(pattern, na=False, regex=True)
        # Map the matches back to the main dataframe
        df.loc[mask.reindex(df.index, fill_value=False), 'macro_sector'] = sector

    # 6. Load to Processed Storage
    logger.info(f"Mapping complete. Saving clean dataset to {output_path}...")
    df.to_csv(output_path, index=False, compression='gzip')
    logger.info("Phase 3 Complete. Pipeline finalized.")

if __name__ == "__main__":
    main()