"""
ExoRisk_PD_Engine: Phase 6 - Interactive Streamlit UI
Author: Hon Seng Choi, Principal Quantitative Architect
Context: A quantitative head-up display demonstrating dynamic credit risk pricing.
         Optimized for a zero-scroll, side-by-side executive dashboard experience.
"""

import streamlit as st
import pandas as pd
import xgboost as xgb
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="ExoRisk Engine", page_icon="🏦", layout="wide")

# --- CSS HACK: REMOVE TOP PADDING FOR ZERO-SCROLL ---
st.markdown("""
    <style>
           .block-container {
                padding-top: 1.5rem;
                padding-bottom: 0rem;
            }
    </style>
    """, unsafe_allow_html=True)

# --- MLOPS: CACHE ENGINE & MATRIX SCHEMA ---
@st.cache_resource
def load_system():
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "models" / "pd_engine_v1.json"
    matrix_path = project_root / "data" / "processed" / "model_matrices" / "X_train.parquet"
    
    # Load XGBoost Engine
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Load data and create a template using the Portfolio Median
    # Median is used instead of Mean to prevent extreme outliers from skewing the baseline
    df_raw = pd.read_parquet(matrix_path)
    template = pd.DataFrame([df_raw.median()])
        
    return model, template

model, df_template = load_system()

# --- TOP LEVEL INFERENCE DISPLAY (OPTIMIZED HUD) ---
header_left, header_right = st.columns([2.5, 1])

with header_left:
    # Centering the subtitle directly under the main title using HTML
    st.markdown("""
        <div style='text-align: center;'>
            <h2 style='margin-bottom: 0px;'>🏦 EXORISK INSTITUTIONAL PRICING ENGINE</h2>
            <p style='color: #A0A0A5; font-size: 15px; margin-top: 5px;'>
                Quantitative Head-Up Display | <b>Strict SHAP-Aligned Parameters</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

with header_right:
    # Live metric placeholder
    metric_placeholder = st.empty()

st.markdown("---")

# --- MASTER UI LAYOUT: SIDE-BY-SIDE ---
# We split the screen: 70% for Micro, 30% for Macro to eliminate scrolling
master_micro, master_macro = st.columns([2.2, 1])

with master_micro:
    st.subheader("Micro: Idiosyncratic Borrower Profile")
    
    # Sub-divide the Micro section into a tight 2-column grid
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        loan_amnt = st.slider("Loan Amount ($)", 1000, 40000, 15000, step=500)
        int_rate = st.slider("Target Interest Rate (%)", 5.0, 36.0, 12.0, step=0.1)
        home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN"])
        term = st.radio("Loan Term (Months)", [36, 60], horizontal=True)

    with col_m2:
        annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=5000)
        dti = st.slider("Debt-to-Income Ratio (DTI)", 1.0, 40.0, 18.0, step=0.5)
        fico = st.slider("FICO Score (Low)", 660, 850, 720, step=5)
        macro_sector = st.selectbox("Employment Sector", [
        "Finance",               # Moved to the top so it defaults automatically
        "Technology", 
        "Logistics_Transport", 
        "Engineering_Science", 
        "Healthcare", 
        "Construction_Trades",
        "Retail_Hospitality", 
        "Manufacturing", 
        "Education", 
        "Government_Public_Sector",
        "Real_Estate", 
        "Energy_Mining", 
        "Media_Entertainment", 
        "Other"
    ])

with master_macro:
    st.subheader("Macro: Economic Regime")
    st.info("Systemic risk adjustment based strictly on the Federal Reserve cost of capital at the time of origination.")
    treasury_yield = st.slider("Treasury 2Y Yield (%)", 0.0, 6.0, 2.5, step=0.1)

# --- REAL-TIME PREDICTION LOGIC ---
X_pred = df_template.copy()

# Map inputs
X_pred['loan_amnt'] = loan_amnt
X_pred['term'] = term
X_pred['int_rate'] = int_rate
X_pred['annual_inc'] = annual_inc
X_pred['dti'] = dti
X_pred['fico_range_low'] = fico
X_pred['Treasury_2Y_Yield'] = treasury_yield

if f"home_ownership_{home_ownership}" in X_pred.columns:
    X_pred[f"home_ownership_{home_ownership}"] = 1.0
if f"macro_sector_{macro_sector}" in X_pred.columns:
    X_pred[f"macro_sector_{macro_sector}"] = 1.0
    
# Inference
pd_score = model.predict_proba(X_pred)[0][1]

# --- UPDATE TOP METRIC ---
if pd_score < 0.15:
    risk_color = "#00FF00" # Green
    tier = "A/B (Prime)"
elif pd_score < 0.30:
    risk_color = "#FFA500" # Orange
    tier = "C/D (Near Prime)"
else:
    risk_color = "#FF0000" # Red
    tier = "E/F/G (Subprime)"

# Inject the result back into the top right placeholder
with metric_placeholder.container():
    st.markdown(f"<h1 style='text-align: right; color: {risk_color}; font-size: 3.5rem; margin-bottom: 0px; padding-bottom: 0px;'>{pd_score * 100:.2f}% PD</h1>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: right; margin-top: 0px; padding-top: 0px;'>Suggested Risk Tier: {tier}</h5>", unsafe_allow_html=True)