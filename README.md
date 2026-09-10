# 🏦 ExoRisk: Institutional Credit Pricing Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B.svg)
![Status](https://img.shields.io/badge/Status-Staging_%2F_UAT-blue.svg)

**Architect:** Hon Seng Choi | Principal Quantitative Architect  
**Target Scope:** Institutional Consumer Credit Portfolio (1.1M+ Records)  
**Core Infrastructure:** Python, XGBoost, Parquet, SHAP, Streamlit  

👉 **[View the Live Interactive HUD (Streamlit Cloud)](https://exorisk-credit-pricing-engine.streamlit.app/)**
*(Quantitative Head-Up Display | Production-Ready Beta)*

## 📌 Executive Summary
Traditional consumer credit models evaluate borrower risk (FICO, Income) in a vacuum, often presenting an opportunity for broader macroeconomic integration. A borrower with a 750 FICO and a $150k income presents a vastly different risk profile in a 5% yield environment compared to a 1% environment—especially when accounting for the volatility of their specific employment sector.

**ExoRisk** is an end-to-end machine learning architecture built specifically to isolate and study this "Sector Delta Risk." By utilizing the **2-Year US Treasury Yield** as a macroeconomic anchor to hold systemic monetary conditions constant, the gradient-boosted engine can accurately measure how default rates diverge across different employment sectors. This allows the system to dynamically price consumer credit based on idiosyncratic borrower fundamentals, sector-specific resilience, and the real-time institutional cost of capital.

To deploy this engine safely to a live institutional underwriting desk, a **closed-form Bayesian calibration layer** was mathematically injected into the pipeline. This corrects the log-odds margin inflation caused by cost-sensitive training, compressing the Brier Score down to **0.1538** while maintaining an elite global ranking power (0.7049 ROC-AUC).

## 📖 Deep Dives & Core Documentation
For a comprehensive breakdown of the macroeconomic thesis, the algorithmic math, and the latent credit anomalies discovered during SHAP analysis, please refer to the core documentation:
* 📄 **[The ExoRisk Business Whitepaper](ExoRisk_Institutional_Credit_Pricing.md)**
* 🛡️ **[SR 11-7 Model Card & Regulatory Governance](MODEL_CARD.md)**

## 🏗️ System Architecture
The repository features a fully idempotent pipeline, transitioning raw tabular data into a production-grade inference application:

1. **Idempotent ETL Pipeline (`/src/data_pipeline/`):** 
   - Processes 1.1M+ rows of LendingClub data (2007-2018).
   - Engineers micro-features and utilizes Regex to map unstructured job titles into 14 distinct macroeconomic sectors.
   - Outputs highly compressed `.parquet` matrices for optimized read/write speeds.
2. **Gradient Boosting Engine (`/src/ml_engine/`):** 
   - Trains an `XGBClassifier` utilizing cost-sensitive learning (`scale_pos_weight`) to handle severe default-class imbalance.
   - Validated via Out-Of-Time (OOT) testing to ensure temporal robustness.
3. **SHAP Explainability (`05_shap_explainer.py`):** 
   - Extracts exact global and local feature importance to align UI parameters strictly with mathematical drivers.
4. **Quantitative Head-Up Display (`underwriting_engine_ui.py`):** 
   - A zero-scroll, institutional-grade Streamlit dashboard. 
   - Implements a **Portfolio Median Baseline** under the hood to prevent "Synthetic Identity" ghost artifacts during UI inference.

```text
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                       EXORISK DUAL-LAYER RISK ARCHITECTURE                       │
 ├──────────────────────────────────────────────────────────────────────────────────┤
 │                                                                                  │
 │  [ MACRO REGIME LAYER ]                   [ MICRO BORROWER LAYER ]               │
 │  ├── FRED API: 2Y Treasury Yield          ├── 1.1M+ LendingClub Records          │
 │  └── Monetary Policy Proxy Signals        └── Debt, FICO, Income, DTI Attributes │
 │                    │                                        │                    │
 │                    └──────────────────┬─────────────────────┘                    │
 │                                       ▼                                          │
 │  [ IDEMPOTENT ETL & STAGING LAYER ]                                              │
 │  ├── Unstructured Job Title NLP / Regex Parser (14 Macro Sectors)                │
 │  ├── Temporal Alignment & Out-of-Time (OOT) Chronological Split                  │
 │  └── Compressed Parquet Matrix Generation (X_train.parquet, X_test.parquet)      │
 │                                       │                                          │
 │                                       ▼                                          │
 │  [ QUANTITATIVE ML CORE ENGINE ]                                                 │
 │  ├── Cost-Sensitive XGBoost Classifier (scale_pos_weight optimization)           │
 │  ├── TreeSHAP Interpretability Vector Extraction (Sector Delta Isolation)        │
 │  └── Decoupled Artifact Export (pd_engine_v1.json + model_meta.json)             │
 │                                       │                                          │
 │                                       ▼                                          │
 │  [ HIGH-THROUGHPUT INFERENCE HUD ] (Streamlit)                                   │
 │  ├── MRM Fail-Fast Protocol: Dynamic JSON Metadata Extraction                    │
 │  ├── Closed-Form Bayesian Calibration: P_cal = p / (p + w(1-p))                  │
 │  └── Dynamic PD Risk Classification: Prime (A/B) vs. Near-Prime vs. Subprime     │
 │                                                                                  │
 └──────────────────────────────────────────────────────────────────────────────────┘

```

## 🖥️ Live Dashboard & Deployment
The UI is built as a highly optimized, interactive terminal. Users can adjust macroeconomic regimes and micro-borrower profiles to witness real-time Probability of Default (PD) recalibrations.

### Local Quick Start
To spin up the ExoRisk Quantitative HUD on your local machine:

```bash
# 1. Clone the repository
git clone https://github.com/honsengchoi1/ExoRisk-PD-Engine.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Quantitative Engine
streamlit run underwriting_engine_ui.py


