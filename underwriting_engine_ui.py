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
import json

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
    # 1. Load the core XGBoost model
    model = xgb.XGBClassifier()
    model_path = Path("models/pd_engine_v1.json")
    model.load_model(model_path)
    
    # 2. Load the lightweight pre-calculated median baseline
    median_path = Path("models/portfolio_median.json")
    import json
    with open(median_path, "r") as f:
        median_dict = json.load(f)
        
    template = pd.DataFrame([median_dict])
    
    return model, template

model, df_template = load_system()

# --- HIGH-PRECISION ENGINEERING CALLBACK LAYER ---
# Initialize default session states
default_states = {
    "loan_amnt_val": 15000,
    "int_rate_val": 12.0,
    "home_ownership_val": "MORTGAGE",
    "term_val": 36,
    "annual_inc_val": 65000,
    "dti_val": 18.0,
    "fico_val": 720,
    "macro_sector_val": "Finance",
    "treasury_yield_val": 2.5
}

for key, value in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

def trigger_param_reset():
    for key, value in default_states.items():
        st.session_state[key] = value

# --- TOP LEVEL INFERENCE DISPLAY (OPTIMIZED HUD) ---
header_left, header_right = st.columns([2.5, 1])

with header_left:
    st.markdown("""
        <div style='text-align: center;'>
            <h2 style='margin-bottom: 0px;'>🏦 EXORISK INSTITUTIONAL PRICING ENGINE</h2>
            <p style='color: #A0A0A5; font-size: 15px; margin-top: 5px;'>
                Quantitative Head-Up Display | <b>Strict SHAP-Aligned Parameters</b>
            </p>
        </div>
    """, unsafe_allow_html=True)

with header_right:
    metric_placeholder = st.empty()

st.markdown("---")

# --- MASTER UI LAYOUT: SIDE-BY-SIDE ---
master_micro, master_macro = st.columns([2.2, 1])

with master_micro:
    st.subheader("Micro: Idiosyncratic Borrower Profile")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        loan_amnt = st.slider("Loan Amount ($)", 1000, 40000, key="loan_amnt_val", step=500)
        int_rate = st.slider("Target Interest Rate (%)", 5.0, 36.0, key="int_rate_val", step=0.1)
        home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "RENT", "OWN"], key="home_ownership_val")
        term = st.radio("Loan Term (Months)", [36, 60], key="term_val", horizontal=True)

    with col_m2:
        annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, key="annual_inc_val", step=5000)
        dti = st.slider("Debt-to-Income Ratio (DTI)", 1.0, 40.0, key="dti_val", step=0.5)
        fico = st.slider("FICO Score (Low)", 660, 850, key="fico_val", step=5)
        macro_sector = st.selectbox("Employment Sector", [
            "Finance", 
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
        ], key="macro_sector_val")

with master_macro:
    st.subheader("Macro: Economic Regime")
    st.info("Systemic risk adjustment based strictly on the Federal Reserve cost of capital at the time of origination.")
    treasury_yield = st.slider("Treasury 2Y Yield (%)", 0.0, 6.0, key="treasury_yield_val", step=0.1)
    
    # --- FULL-WIDTH RESET BUTTON WITH CALLBACK ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("🔄 Reset Sliders", on_click=trigger_param_reset, use_container_width=True)

# --- REAL-TIME PREDICTION LOGIC ---
X_pred = df_template.copy()

# Map inputs from session state variables
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
    
# --- INFERENCE & CALIBRATION ---
pd_raw = model.predict_proba(X_pred)[0][1]

try:
    meta_path = Path(__file__).resolve().parent / "models" / "model_meta.json"
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    dynamic_weight = float(metadata['scale_pos_weight'])
except (FileNotFoundError, KeyError):
    st.error("🚨 CRITICAL MRM HALT: 'model_meta.json' metadata missing. Inference halted to prevent uncalibrated pricing.")
    st.stop()

# Real-time institutional calibration (Bayesian correction)
pd_score = pd_raw / (pd_raw + dynamic_weight * (1.0 - pd_raw))

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