# project_state.md
**Last Updated:** 2026-09-08
**Project:** ExoRisk Codebase: Phase 3 Feature Merge Complete
**Phase:** 4.0 - Machine Learning Engine (XGBoost) Initialization

## ⚠️ Permanent Executive Directives
1. **Verification-First Workflow:** Always brainstorm, discuss, teach, learn, confirm, and proceed.
2. **Zero Unauthorized Execution:** Do NOT generate code until explicitly requested by the user.
3. **Career Pivot Integration:** Continuously evaluate how architectural decisions map to my "Flattened Executive" resume strategy and IC quant risk positioning.

## Executive Pitch (For README/Whitepaper)
"I engineered an end-to-end XGBoost predictive underwriting pipeline that dynamically prices consumer credit risk. By normalizing asynchronous macroeconomic indicators (sector job momentum, treasury yields) into a multi-index Parquet panel and merging it with micro-level borrower data, the machine learning engine calculates a highly calibrated Probability of Default (PD). This ensures the model understands the exact economic environment and sector volatility at the exact moment a loan is originated, bridging the gap between macro-regime shifts and idiosyncratic consumer risk."

## Architectural Standards (For README)
* **Strict Python Typing:** Enforced via Pylance to create executable, built-in documentation. This guarantees data type integrity, eliminating runtime friction during MLOps deployment hand-offs.
* **Idempotent Data Engineering:** Constructed highly resilient ETL pipelines that bypass redundant processing, optimizing disk I/O and RAM allocation during massive dataset transformation.

## End-to-End Pipeline Roadmap
1.  **[COMPLETED] Macro Ingestion:** Vectorized API extraction of FRED data.
2.  **[COMPLETED] Micro Ingestion:** Idempotent extraction of 1.26GB LendingClub dataset.
3.  **[COMPLETED] Feature Assembly:** Vectorized Regex mapping of 2.2 million job titles into 14 macro sectors; generated merged, ML-ready payload.
4.  **[CURRENT] Engine Training:** Train XGBoost on the unified matrix.
5.  **Explainability & UI:** Generate SHAP values; wrap serialized model in an interactive Streamlit UI.

## Future Pipeline Expansions
* **Phase X: Unsupervised Macro Clustering:** Apply K-Means to the macro feature matrix to isolate discrete "Economic Regimes". Inject these Regime IDs into the XGBoost engine as categorical features to dynamically alter underwriting logic.