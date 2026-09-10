# ExoRisk Predictive Underwriting Engine
**Last Updated:** 2026-09-10
**Project:** ExoRisk Codebase: Phase 6 Final Deployment
**Phase:** 6.0 - Production Launch, Bayesian Calibration & MRM Governance [COMPLETED]

## Executive Overview
ExoRisk is an end-to-end XGBoost predictive underwriting pipeline designed to dynamically price consumer credit risk. By normalizing asynchronous macroeconomic indicators (Treasury Yields, Sector Job Momentum) into a multi-index Parquet panel and merging it with micro-level borrower data, the machine learning engine calculates a highly calibrated Probability of Default (PD). 

To ensure safe deployment to an institutional underwriting desk, the architecture utilizes a closed-form Bayesian calibration layer to correct log-odds margin inflation (compressing the Brier Score to 0.1538). This ensures the model understands the specific economic environment at the exact moment a loan is originated, bridging the gap between macro-regime shifts and idiosyncratic consumer risk.

## Architectural Standards
* **Strict Python Typing:** Enforced to create self-documenting code and guarantee data type integrity, eliminating runtime friction during MLOps deployment hand-offs.
* **Idempotent Data Engineering:** Resilient ETL pipelines designed to bypass redundant processing, optimizing disk I/O and RAM allocation during the transformation of 1.1M+ row datasets.
* **Decoupled MRM Governance:** Explainer artifacts (TreeSHAP) and dynamic model metadata (`model_meta.json`) are decoupled from the core training loop to bypass serialization bugs and enable strict "Fail-Fast" safeguards during UI inference.

## End-to-End Pipeline Roadmap
1.  **[COMPLETED] Macro Ingestion:** Vectorized API extraction of Federal Reserve (FRED) economic data.
2.  **[COMPLETED] Micro Ingestion:** Idempotent extraction of 1.1M+ terminal LendingClub borrower records.
3.  **[COMPLETED] Feature Assembly:** Vectorized Regex mapping of unstructured job titles into 14 distinct macro-economic sectors.
4.  **[COMPLETED] Engine Training:** XGBoost Classifier trained on a unified, time-aligned feature matrix, utilizing cost-sensitive learning (4.08 weight) to handle class imbalance.
5.  **[COMPLETED] Explainability & Calibration:** Extracted Shapley values to validate Sector Delta Risk; applied Bayesian probability calibration for production accuracy.
6.  **[COMPLETED] Interactive UI:** Wrapped the serialized JSON model, baseline telemetry, and metadata into a zero-scroll, institutional Streamlit HUD.

## Future Pipeline Expansions
* **Unsupervised Macro Clustering:** Apply K-Means clustering to the macroeconomic feature matrix to mathematically define discrete "Economic Regimes" (e.g., Expansion, Contraction, Stagflation) as standalone categorical features.
* **Live LOS Integration:** Transition the Streamlit UI into a headless FastAPI microservice to ingest real-time JSON payloads from a live Loan Origination System (LOS).