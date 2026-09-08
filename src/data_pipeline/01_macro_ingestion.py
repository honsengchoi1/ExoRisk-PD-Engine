"""
ExoRisk_PD_Engine: Phase 1 Macro ETL Ingestion
Author: Hon Seng Choi, Principal Quantitative Architect
Context: Pulls macroeconomic time-series (FRED API), calculates momentum (lag horizons), 
         and structures a flat MultiIndex panel for XGBoost ingestion.
Output: Snappy-compressed Parquet (Date, Sector).
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from fredapi import Fred

# Configure Executive Telemetry Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MacroETL")

def get_project_root() -> Path:
    """Dynamically resolves the project root based on this script's location."""
    # Assuming script is in src/data_pipeline/01_macro_ingestion.py
    return Path(__file__).resolve().parents[2]

def extract_fred_data(api_key: str, targets: Dict[str, str]) -> pd.DataFrame:
    """Extracts and frequency-aligns FRED time series to Month-End (ME)."""
    fred = Fred(api_key=api_key)
    raw_series = []

    for ticker, name in targets.items():
        try:
            # Downsample to ME using last available observation
            s = fred.get_series(ticker).dropna().resample('ME').last()
            s.name = name
            raw_series.append(s)
            logger.info(f"Extracted: {ticker} -> {name}")
        except Exception as e:
            logger.error(f"Extraction failed for {ticker}: {e}")
            sys.exit(1)

    # Concat wide, forward-fill async publication gaps
    df_wide = pd.concat(raw_series, axis=1).sort_index()
    return df_wide.ffill()

def main() -> None:
    start_time = time.perf_counter()
    logger.info("Initiating Phase 1: Macro ETL Ingestion")

    # 1. Credential & Path Management
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        logger.error("FATAL: FRED_API_KEY environment variable not detected.")
        sys.exit(1)

    project_root = get_project_root()
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 2. Target Series Specification
    macro_series: Dict[str, str] = {'DGS2': 'Treasury_2Y_Yield'}
    sector_series: Dict[str, str] = {
        'USINFO': 'Information',
        'USEHS': 'Education_Health',
        'USCONS': 'Construction',
        'USFIRE': 'Financial',
        'USTRADE': 'Trade_Retail'
    }
    all_targets = {**macro_series, **sector_series}

    # 3. Extraction
    df_wide = extract_fred_data(api_key, all_targets)
    # Drop rows where payroll data hasn't started yet
    df_wide = df_wide.dropna(subset=list(sector_series.values()), how='all')

    # 4. Feature Engineering: Lag Horizon Computation
    logger.info("Computing vectorized 1M, 3M, and 6M rolling momentum...")
    horizons: List[int] = [1, 3, 6]
    
    for sector in sector_series.values():
        for lag in horizons:
            df_wide[f'{sector}_{lag}M_Delta'] = df_wide[sector].pct_change(periods=lag)

    # Clean leading NaNs caused by the 6M rolling window
    df_wide = df_wide.dropna()

    # 5. Schema Transformation: Melt to MultiIndex Panel
    logger.info("Executing wide-to-flat schema transformation...")
    melted_dfs = []
    
    for sector in sector_series.values():
        sector_cols = [
            sector, 
            f'{sector}_1M_Delta', 
            f'{sector}_3M_Delta', 
            f'{sector}_6M_Delta', 
            'Treasury_2Y_Yield'
        ]
        
        df_sector = df_wide[sector_cols].copy()
        df_sector = df_sector.rename(columns={
            sector: 'Payroll_Level',
            f'{sector}_1M_Delta': '1M_Pct_Delta',
            f'{sector}_3M_Delta': '3M_Pct_Delta',
            f'{sector}_6M_Delta': '6M_Pct_Delta'
        })
        
        df_sector['Sector'] = sector
        df_sector = df_sector.reset_index(names='Date')
        melted_dfs.append(df_sector)

    df_flat = pd.concat(melted_dfs, ignore_index=True)
    df_flat = df_flat.set_index(['Date', 'Sector']).sort_index()

    # 6. Serialization & Telemetry
    out_path = interim_dir / "macro_features_base.parquet"
    
    try:
        df_flat.to_parquet(out_path, compression='snappy')
        
        # Calculate memory footprint for telemetry
        mem_mb = df_flat.memory_usage(deep=True).sum() / (1024 ** 2)
        exec_time = time.perf_counter() - start_time
        
        logger.info(f"ETL Complete: {df_flat.shape[0]} rows | {df_flat.shape[1]} columns")
        logger.info(f"Memory Footprint: {mem_mb:.2f} MB")
        logger.info(f"Output serialized to: {out_path}")
        logger.info(f"Execution Time: {exec_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"FATAL: Serialization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()