# ExoRisk: SR 11-7 Model Governance Card

## 1. Intended Use & Methodological Core
*   **Primary Objective:** To quantify "Sector Delta Risk" by mapping macroeconomic labor shifts to consumer credit default rates. The model was specifically architected to stress-test sector divergences, such as the August Nonfarm Payrolls (NFP) report demonstrating job expansions in Government and Healthcare alongside contractions in the Information sector.
*   **Macro Anchor Proxy:** Instead of introducing excessive noise through dozens of collinear macroeconomic variables, the engine utilizes the 2-Year US Treasury Yield (`Treasury_2Y_Yield`) as a singular, efficient proxy. The 2-Year yield natively prices in systemic monetary conditions, inflation expectations, and economic sentiment, holding the broader macro environment constant to isolate sector-specific idiosyncratic risk.
*   **Validation Cohort Strategy:** The engine enforces a strict Out-of-Time (OOT) temporal split to evaluate forward-looking predictive stability and prevent temporal data leakage:
    *   **In-Time Development Matrix (Train):** 2007-01-01 to 2016-12-31 ($N = 1,119,699$ terminal loans)
    *   **Out-of-Time Stress Matrix (Test):** 2017-01-01 to 2018-12-31

---

## 2. Out-of-Time (OOT) Performance & Validation Metrics
Evaluated strictly against unseen 2017–2018 loan originations:

| Metric | Score | Risk Management Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.7049** | Strong global discrimination and borrower ranking across differing macro regimes. |
| **PR-AUC** | **0.3747** | High minority-class precision relative to the baseline unconditional default rate. |
| **Brier Score** | **0.1538** | Calibrated quadratic probability accuracy reflecting the post-Bayesian correction. |

---

## 3. Algorithmic Architecture & Mathematical Governance

### A. Cost-Sensitive Class Balancing
To mitigate an empirical class imbalance ratio of **4.08 : 1** without introducing synthetic sampling artifacts (e.g., SMOTE distortion), the engine injects a native cost-sensitive multiplier into the gradient computation:

$$scale\_pos\_weight = \frac{N_{\text{negative}}}{N_{\text{positive}}} = 4.08$$

This scales the loss gradient for minority default instances ($y = 1$), ensuring the tree splits prioritize default discrimination over majority class accuracy.

### B. Closed-Form Bayesian Calibration
While the cost-sensitive multiplier achieves high ranking power, the raw sigmoidal output ($P_{\text{raw}}$) suffers from severe log-odds margin inflation. To deploy this to a live underwriting environment, a closed-form Bayesian probability calibration layer is mathematically injected into the inference pipeline:

$$P_{\text{calibrated}} = \frac{P_{\text{raw}}}{P_{\text{raw}} + w(1 - P_{\text{raw}})}$$

This restores the true portfolio odds ratio, compressing the Brier Score from 0.2477 down to 0.1538.

### C. Structural Multi-Collinearity Immunity
Unlike classical Ordinary Least Squares (OLS) or multi-factor logistic regressions, gradient boosted decision trees (GBDT) do not invert covariance matrices. The architecture natively handles correlated inputs, allowing concurrent ingestion of 1M, 3M, and 6M rolling sector payroll momentum deltas without variance inflation or coefficient instability.

---

## 4. Model Interpretability & Explainability (SR 11-7 Attribution)
To satisfy regulatory transparency requirements regarding black-box models, local and global feature attribution was conducted using Shapley Additive Explanations (**TreeSHAP**) across a representative sample of 5,000 OOT observations.

### A. Global Feature Hierarchy
*   **Dominant Macro Proxy:** `Treasury_2Y_Yield` established itself as the **#2 most impactful feature** in the global architecture, surpassed only by loan interest rate (`int_rate`). High 2-Year Treasury yields exert a strong positive SHAP impact, directly elevating predicted PD across borrower grades.
*   **Core Micro Governance:** Traditional underwriting variables (`term`, `fico_range_low`, `dti`, `loan_amnt`) preserve expected monotonic relationships: higher FICO and income compress default odds, while extended terms (60-Month) increase baseline risk.

### B. Sector Delta Risk Attribution
*   **Risk-Suppressing Sectors:** `macro_sector_Technology` and `macro_sector_Finance` consistently produced negative SHAP values, dampening default probability and serving as structural credit buffers.
*   **Risk-Amplifying Sectors:** `macro_sector_Logistics_Transport` and `macro_sector_Hospitality_Food` produced positive SHAP values, driving default risk upward.
*   **Latent Contract Extraction:** The engine autonomously mapped empirical risk divergences between stable salaried earners (Healthcare, Education, Government) and volatile wage/variable-income profiles without explicit employment-type labeling.

### C. Architectural Separation of Concerns
Due to known serialization bugs in XGBoost 2.x exporting leaf float values as bracketed strings, standard TreeExplainer parsers fail at inference. To protect the core pipeline, ExoRisk enforces strict separation of concerns: SHAP telemetry is decoupled into an isolated diagnostic script (`05_shap_explainer.py`), ensuring the production engine serializes flawlessly without brittle namespace injections.

---

## 5. Boundary Conditions, Extrapolation Limits & Out-of-Scope Use

### A. Tree-Based Upper-Bound Ceiling (The 4.87% Threshold)
*   **The Quantitative Constraint:** Decision trees split feature spaces into orthogonal step functions; they cannot extrapolate linearly beyond empirical sample boundaries. The maximum 2-Year Treasury Yield observed within the training matrix is **4.87%** (Q3 2007).
*   **Anomalous High-Yield Behavior:** At hypothetical yields approaching or exceeding 5.00%, the inference engine routes vectors into the outermost terminal leaves, causing predicted PD to artificially flatline or display regime-inversion anomalies.
*   **Mandated MRM Governance Remediation:**
    1.  **Input Clipping:** Live pipelines must enforce hard bounding locks, clipping inputs at the historical 99th percentile ($Yield_{2Y} \le 4.87\%$).
    2.  **Model Retraining Protocol:** Portfolios operating under a persistent 5%+ Federal Reserve rate environment must incorporate 2022–2024 origination matrices to partition discrete high-cost-of-capital leaf topologies.

### B. Right-Censoring Delimitation
To prevent right-censoring bias, the dataset strictly isolates terminal outcomes (`Fully Paid` vs. `Charged Off`). Consequently, this engine is strictly an **Origination Pricing Model** and is out-of-scope for ongoing lifetime Expected Credit Loss (ECL) or transition-matrix modeling under CECL / IFRS 9 without dynamic survival-analysis adjustments.

### C. Institutional Deployment Delta
The accompanying Streamlit UI serves as a deterministic pricing head-up display (HUD). Production deployment into an active enterprise environment requires integration with a live Loan Origination System (LOS), automated decision-engine gateways, and nightly batch processing for macro-regime state updates.