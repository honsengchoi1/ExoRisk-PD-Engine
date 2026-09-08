# project_state.md
**Last Updated:** 2026-09-08
**Project:** ExoRisk Codebase: Phase 2 Micro Ingestion & NLP Regex
**Phase:** 2.1 - Data Procurement & Jupyter Scratchpad Initialization

## ⚠️ Permanent Executive Directives
1. **Verification-First Workflow:** Always brainstorm, discuss, teach, learn, confirm, and proceed.
2. **Zero Unauthorized Execution:** Do NOT generate code until explicitly requested by the user.
3. **Career Pivot Integration:** Continuously evaluate how architectural decisions map to my "Flattened Executive" resume strategy and IC quant risk positioning.

## Executive Pitch (For README/Whitepaper)
"I engineered an end-to-end XGBoost predictive underwriting pipeline that dynamically prices consumer credit risk. By normalizing asynchronous macroeconomic indicators (sector job momentum, treasury yields) into a multi-index Parquet panel and merging it with micro-level borrower data, the machine learning engine calculates a highly calibrated Probability of Default (PD). This ensures the model understands the exact economic environment and sector volatility at the exact moment a loan is originated, bridging the gap between macro-regime shifts and idiosyncratic consumer risk."

## Architectural Standards (For README)
* **Strict Python Typing:** Enforced via Pylance to create executable, built-in documentation. This guarantees data type integrity, eliminating runtime friction during MLOps deployment hand-offs.
* **Parquet Serialization:** Converted raw, massive CSV payloads into columnar, Snappy-compressed Parquet matrices, achieving a 90%+ reduction in disk I/O and memory overhead during XGBoost training.

## End-to-End Pipeline Roadmap
1.  **[COMPLETED] Macro Ingestion:** Vectorized API extraction of FRED employment and yield data. Computed 1M/3M/6M rolling momentum features. Flattened into a multi-index Parquet panel.
2.  **[CURRENT] Micro Ingestion:** Convert LendingClub CSV to Parquet; apply Regex Job-to-Sector mapping.
3.  **Feature Assembly:** Left join micro data with the macro matrix based on (Lagged_Date, State, Sector).
4.  **Engine Training:** Train XGBoost on the unified matrix.
5.  **Explainability & UI:** Generate SHAP values; wrap serialized model in an interactive Streamlit UI.

## Future Pipeline Expansions
* **Phase X: Unsupervised Macro Clustering:** Apply K-Means to the macro feature matrix to isolate discrete "Economic Regimes". Inject these Regime IDs into the XGBoost engine as categorical features to dynamically alter underwriting logic.