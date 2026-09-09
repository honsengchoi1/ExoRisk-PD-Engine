"""
ExoRisk_PD_Engine: Phase 4a - Model Data Prep & Temporal Split
Author: Hon Seng Choi, Principal Quantitative Architect
Context: Aggregates micro features, executes key join with macro lookup, merges Phase 1 
         FRED economic indicators, and executes Out-of-Time (OOT) temporal split.
Output: X_train, X_test, y_train, y_test (Snappy-compressed Parquet).
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Configure Executive Telemetry Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataPrep_OOT")

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def main() -> None:
    logger.info("Initiating Phase 4a: Model Data Preparation & OOT Split (Integrity Guaranteed)")

    # 1. Path Routing & Idempotent Gatekeeper
    project_root = get_project_root()
    raw_path = project_root / "data" / "raw" / "accepted_2007_to_2018Q4.csv.gz"
    phase3_path = project_root / "data" / "processed" / "macro_sector_lookup.parquet"
    
    # NEW PATH: Phase 1 FRED Macroeconomic Data
    phase1_path = project_root / "data" / "interim" / "macro_features_base.parquet"
    
    model_data_dir = project_root / "data" / "processed" / "model_matrices"
    
    model_data_dir.mkdir(parents=True, exist_ok=True)
    check_file = model_data_dir / "X_train.parquet"
    
    if check_file.exists():
        logger.info(f"Model matrices already exist at {model_data_dir}. Bypassing Phase 4a.")
        sys.exit(0)

    if not raw_path.exists() or not phase3_path.exists() or not phase1_path.exists():
        logger.error("Prerequisite data missing. Ensure Phases 1, 2, and 3 are complete.")
        sys.exit(1)

    # 2. Comprehensive Data Extraction
    logger.info("Loading Micro features, Phase 3 Lookup, and Phase 1 Macro Environment...")
    
    micro_cols = [
        'id', 'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 
        'emp_length', 'home_ownership', 'annual_inc', 'issue_d', 
        'loan_status', 'purpose', 'dti', 'fico_range_low'
    ]
    
    df_raw = pd.read_csv(raw_path, usecols=micro_cols, low_memory=False, compression='gzip')
    df_lookup = pd.read_parquet(phase3_path)
    
    # Load FRED Data and reset index to expose 'Date' and 'Sector' for merging
    df_fred = pd.read_parquet(phase1_path).reset_index()

    # 3. Dimensional Integrity Merges
    logger.info("Executing deterministic key-joins to build unified feature space...")
    
    # Merge 1: Attach Macro Sector to Micro Profile
    df = pd.merge(df_raw, df_lookup, on='id', how='left', validate='1:1')
    df = df.drop(columns=['id'])

    # Format Origination Date for Temporal Merging
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    
    # Align micro origination date (1st of month) to FRED reporting date (Month-End)
    df['macro_date'] = df['issue_d'] + pd.offsets.MonthEnd(0)

    # Merge 2: Attach Phase 1 FRED Macro Indicators based on Date and Sector
    df = pd.merge(
        df,
        df_fred,
        left_on=['macro_date', 'macro_sector'],
        right_on=['Date', 'Sector'],
        how='left'
    )

    # Impute missing macro features (e.g., for the 'Other' sector)
    # Treasury Yield is the same for all sectors in a given month
    df['Treasury_2Y_Yield'] = df.groupby('macro_date')['Treasury_2Y_Yield'].transform(lambda x: x.fillna(x.median()))
    
    # Assume 0.0 (neutral) job momentum for unclassified sectors
    for col in ['1M_Pct_Delta', '3M_Pct_Delta', '6M_Pct_Delta']:
        df[col] = df[col].fillna(0.0)

    # Drop redundant join keys to keep matrix clean
    df = df.drop(columns=['Date', 'Sector', 'macro_date', 'Payroll_Level'])

    # 4. Target Formulation (Right-Censoring Mitigation)
    logger.info("Isolating terminal cohorts to eliminate right-censoring bias...")
    terminal_states = ['Fully Paid', 'Charged Off']
    df = df[df['loan_status'].isin(terminal_states)].copy()
    
    df['target_default'] = (df['loan_status'] == 'Charged Off').astype(int)
    
    logger.info(f"Terminal Cohort Size: {len(df):,} records.")
    logger.info(f"Baseline Portfolio Default Rate: {df['target_default'].mean() * 100:.2f}%")

    # 5. Feature Engineering & Cleaning
    logger.info("Formatting datatypes and handling missing micro values...")
    
    df['emp_length'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
    df['emp_length'] = df['emp_length'].fillna(0) 
    
    df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
    
    for col in ['annual_inc', 'dti', 'fico_range_low', 'Treasury_2Y_Yield']:
        df[col] = df[col].fillna(df[col].median())

    # 6. Out-Of-Time (OOT) Temporal Split
    logger.info("Executing Out-Of-Time validation split (Train: <=2016 | Test: >=2017)...")
    
    df = df.sort_values('issue_d')
    
    train_mask = df['issue_d'] < '2017-01-01'
    test_mask = df['issue_d'] >= '2017-01-01'
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # 7. Leakage Prevention & Matrix Finalization
    drop_cols = ['loan_status', 'issue_d', 'target_default']
    
    y_train = train_df['target_default']
    y_test = test_df['target_default']
    
    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    logger.info("One-Hot Encoding categorical features...")
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # 8. Serialization
    logger.info("Serializing finalized design matrices to Parquet...")
    
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    X_train.to_parquet(model_data_dir / "X_train.parquet", compression='snappy')
    X_test.to_parquet(model_data_dir / "X_test.parquet", compression='snappy')
    
    y_train.to_frame().to_parquet(model_data_dir / "y_train.parquet", compression='snappy')
    y_test.to_frame().to_parquet(model_data_dir / "y_test.parquet", compression='snappy')

    logger.info(f"Phase 4a Complete. Train shape: {X_train.shape} | Test shape: {X_test.shape}")

if __name__ == "__main__":
    main()