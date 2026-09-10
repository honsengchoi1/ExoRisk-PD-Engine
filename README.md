# 🏦 ExoRisk: Institutional Credit Pricing Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B.svg)
![Status](https://img.shields.io/badge/Status-Production_Ready-success.svg)

## 📌 Executive Summary
Traditional consumer credit models evaluate borrower risk (FICO, Income) in a vacuum, often presenting an opportunity for broader macroeconomic integration. A borrower with a 750 FICO and a $150k income presents a vastly different risk profile in a 5% yield environment compared to a 1% environment—especially when accounting for the volatility of their specific employment sector.

**ExoRisk** is an end-to-end machine learning architecture built specifically to isolate and study this "Sector Delta Risk." By utilizing the **2-Year US Treasury Yield** as a macroeconomic anchor to hold systemic monetary conditions constant, the gradient-boosted engine can accurately measure how default rates diverge across different employment sectors. This allows the system to dynamically price consumer credit based on idiosyncratic borrower fundamentals, sector-specific resilience, and the real-time institutional cost of capital.

## 🏗️ System Architecture
The repository features a fully idempotent pipeline, transitioning raw tabular data into a production-grade inference application:

1. **Idempotent ETL Pipeline (`/src/data_pipeline/`):** 
   - Processes 2.2M+ rows of LendingClub data (2007-2018).
   - Engineers micro-features and utilizes Regex to map 2.2M unstructured job titles into 14 distinct macroeconomic sectors.
   - Outputs highly compressed `.parquet` matrices for optimized read/write speeds.
2. **Gradient Boosting Engine (`/src/ml_engine/`):** 
   - Trains an `XGBClassifier` utilizing cost-sensitive learning (`scale_pos_weight`) to handle severe default-class imbalance.
   - Validated via Out-Of-Time (OOT) testing to ensure temporal robustness.
3. **SHAP Explainability (`05_shap_explainer.py`):** 
   - Extracts exact global and local feature importance to align UI parameters strictly with mathematical drivers.
4. **Quantitative Head-Up Display (`underwriting_engine_ui.py`):** 
   - A zero-scroll, institutional-grade Streamlit dashboard. 
   - Implements a **Portfolio Median Baseline** under the hood to prevent "Synthetic Identity" ghost artifacts during UI inference.

## 🖥️ Live Dashboard & Deployment
The UI is built as a highly optimized, interactive terminal. Users can adjust macroeconomic regimes and micro-borrower profiles to witness real-time Probability of Default (PD) recalibrations.

## 3. System Architecture

```text
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                       EXORISK DUAL-LAYER RISK ARCHITECTURE                       │
 ├──────────────────────────────────────────────────────────────────────────────────┤
 │                                                                                  │
 │  [ MACRO REGIME LAYER ]                   [ MICRO BORROWER LAYER ]               │
 │  ├── FRED API: 2Y Treasury Yield          ├── 2.2M+ LendingClub Records (07-18)  │
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
 │  ├── TreeSHAP Interpretability Vector Extraction                                 │
 │  └── Serialized Production Artifact Export (pd_engine_v1.json)                   │
 │                                       │                                          │
 │                                       ▼                                          │
 │  [ HIGH-THROUGHPUT INFERENCE HUD ] (Streamlit)                                   │
 │  ├── Memory-Mapped Baseline Injection: median(X_train)                           │
 │  ├── Asymmetric Master HUD (Zero-Scroll Executive UI)                            │
 │  └── Dynamic PD Risk Classification: Prime (A/B) vs. Near-Prime vs. Subprime     │
 │                                                                                  │
 └──────────────────────────────────────────────────────────────────────────────────┘

### Local Quick Start
To spin up the ExoRisk Quantitative HUD on your local machine:

```bash
# 1. Clone the repository
git clone [https://github.com/honsengchoi1/ExoRisk-PD-Engine.git](https://github.com/honsengchoi1/ExoRisk-PD-Engine.git)
cd ExoRisk-PD-Engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Quantitative Engine
streamlit run underwriting_engine_ui.py