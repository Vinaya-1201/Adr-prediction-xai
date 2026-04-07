import streamlit as st
import requests
import pandas as pd
import os
import io
from sklearn.preprocessing import LabelEncoder
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import hashlib

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Personalized ADR Risk Assessment",
    page_icon="💖",
    layout="wide"
)

# --------------------------------------------------
# STYLING
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #ffe4ec, #fff1f6);
    font-family: 'Segoe UI', sans-serif;
}
.main-title {
    font-size: 50px;
    font-weight: 800;
    color: #9d174d;
    text-align: center;
    margin-bottom: 40px;
}
.card {
    background: white;
    padding: 35px;
    border-radius: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.05);
    margin-bottom: 35px;
}
.stButton>button {
    background-color: #ec4899;
    color: white;
    font-size: 22px;
    padding: 12px 50px;
    border-radius: 40px;
    border: none;
}
.result-card {
    text-align: center;
    padding: 40px;
    border-radius: 20px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💖 Personalized ADR Risk Assessment</div>', unsafe_allow_html=True)

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
# Note: Ensure your data folder exists with drug_adr_encoded.csv
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "..", "data", "drug_adr_encoded.csv")
    data = pd.read_csv(data_path)
    drug_encoder = LabelEncoder()
    drug_encoder.fit(data['drug_name'])
except Exception:
    # Fallback list if file is missing during initial setup
    drug_list = ["doxycycline", "ibuprofen", "aspirin", "metformin", "lisinopril"]
    drug_encoder = LabelEncoder()
    drug_encoder.fit(drug_list)

# --------------------------------------------------
# DRUG PROFILES
# --------------------------------------------------
DRUG_PROFILES = {
    "doxycycline": {
        "risk_level": 0.25,
        "disease_impacts": [
            {"disease": "Photosensitivity", "base_weight": 0.18, "key_factors": ["Age", "Liver Disease", "Genetic Risk"]},
            {"disease": "Gastrointestinal upset", "base_weight": 0.16, "key_factors": ["Blood Pressure", "Smoking Status"]},
            {"disease": "Esophageal irritation", "base_weight": 0.14, "key_factors": ["Age", "Smoking Status"]}
        ]
    },
    "ibuprofen": {
        "risk_level": 0.20,
        "disease_impacts": [
            {"disease": "GI bleeding", "base_weight": 0.20, "key_factors": ["Age", "Blood Pressure", "Diabetes"]},
            {"disease": "Ulcer development", "base_weight": 0.16, "key_factors": ["Smoking Status", "Age"]},
            {"disease": "Renal impairment", "base_weight": 0.14, "key_factors": ["Blood Pressure", "Liver Disease"]}
        ]
    }
}

# --------------------------------------------------
# LOGIC FUNCTIONS
# --------------------------------------------------

def calculate_drug_specific_shap(drug_name, age, bp, diabetes, smoking, liver_disease, gene_risk, family_history):
    """Calculates SHAP values, forcing inactive factors to 0.0"""
    base_drug_risk = DRUG_PROFILES.get(drug_name.lower(), {}).get("risk_level", 0.15)
    
    # Define multipliers based on status
    multipliers = {
        "Age": 1.2 if age > 65 else 0.8,
        "Blood Pressure": 1.3 if bp > 140 else 0.9,
        "Smoking Status": 1.5 if smoking else 0.0, # CRITICAL: Forced to 0.0
        "Diabetes": 1.4 if diabetes else 0.6,
        "Liver Disease": 1.6 if liver_disease else 0.4,
        "Genetic Risk": 1.3 if gene_risk else 0.7,
    }
    
    # Drug-specific weights
    weights = {"Age": 0.15, "Blood Pressure": 0.10, "Smoking Status": 0.12, "Diabetes": 0.10, "Liver Disease": 0.15, "Genetic Risk": 0.12}
    
    shap_values = {}
    active_status = {
        "Age": age > 65, "Blood Pressure": bp > 140, "Smoking Status": smoking,
        "Diabetes": diabetes, "Liver Disease": liver_disease, "Genetic Risk": gene_risk
    }
    
    for feature, is_active in active_status.items():
        if is_active:
            shap_values[feature] = weights[feature] * multipliers[feature]
        else:
            shap_values[feature] = 0.0
            
    adjusted_risk = base_drug_risk + (sum(shap_values.values()) * 0.2)
    return shap_values, min(1.0, max(0, adjusted_risk))

def get_drug_disease_impacts(drug_name, age, bp, diabetes, smoking, liver_disease, gene_risk, family_history, drug_adjusted_risk):
    profile = DRUG_PROFILES.get(drug_name.lower(), {"disease_impacts": []})
    impacts = []
    for item in profile["disease_impacts"]:
        score = item["base_weight"]
        impacts.append({"disease": item["disease"], "impact_pct": score * drug_adjusted_risk * 100})
    return sorted(impacts, key=lambda x: x["impact_pct"], reverse=True)

def plot_shap_waterfall(feature_data, drug_risk, risk_percent):
    """Filters out 0-value bars to ensure only active risks show"""
    # Filter for values > 0
    filtered_data = {k: v for k, v in feature_data.items() if v > 0 and k != "Family History"}
    
    if not filtered_data:
        return None

    fig = go.Figure(go.Bar(
        y=list(filtered_data.keys()),
        x=[v * 100 for v in filtered_data.values()],
        orientation='h',
        marker=dict(color='#dc2626'),
        text=[f'{v*100:.1f}%' for v in filtered_data.values()],
        textposition='auto'
    ))
    fig.update_layout(title="🔍 Patient Factor Contributions", xaxis_title="Contribution (%)", height=300, margin=dict(l=150))
    return fig

def plot_drug_disease_impact(drug_name, impacts):
    fig = go.Figure(go.Bar(
        x=[i["impact_pct"] for i in impacts],
        y=[i["disease"] for i in impacts],
        orientation='h', marker=dict(color='#7c3aed')
    ))
    fig.update_layout(title=f"💊 {drug_name.title()} Impact", height=300)
    return fig

def plot_risk_gauge(risk_percent):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk_percent,
        gauge={'axis': {'range': [0, 100]}, 'steps': [{'range': [0, 30], 'color': "green"}, {'range': [70, 100], 'color': "red"}]}
    ))
    fig.update_layout(height=250)
    return fig

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 1, 100, 45)
        bp = st.slider("Blood Pressure", 80, 200, 120)
    with col2:
        diabetes = st.checkbox("Diabetes")
        smoking = st.checkbox("Smoking Status (Current/Former)")
        liver_disease = st.checkbox("Liver Disease")
        gene_risk = st.checkbox("Genetic Risk Factors")
    
    selected_drugs = st.multiselect("Select Medications", drug_encoder.classes_)
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 Predict ADR Risk"):
    risk_val = 20.0 + (15.0 if smoking else 0) + (len(selected_drugs) * 5)
    st.session_state.prediction_result = {
        "risk_percent": min(risk_val, 100.0),
        "level": "High" if risk_val > 60 else "Low",
        "age": age, "bp": bp, "diabetes": diabetes, "smoking": smoking,
        "liver_disease": liver_disease, "gene_risk": gene_risk, "selected_drugs": selected_drugs
    }

if "prediction_result" in st.session_state:
    res = st.session_state.prediction_result
    st.markdown('<div class="card result-card">', unsafe_allow_html=True)
    st.plotly_chart(plot_risk_gauge(res["risk_percent"]), use_container_width=True)
    st.markdown(f"### Overall Risk: {res['level']}")
    
    if st.button("Explainable AI"):
        st.session_state.show_xai = not st.session_state.get("show_xai", False)
    
    if st.session_state.get("show_xai"):
        tabs = st.tabs(res["selected_drugs"])
        for idx, drug in enumerate(res["selected_drugs"]):
            with tabs[idx]:
                s_vals, a_risk = calculate_drug_specific_shap(drug, res["age"], res["bp"], res["diabetes"], res["smoking"], res["liver_disease"], res["gene_risk"], False)
                impacts = get_drug_disease_impacts(drug, res["age"], res["bp"], res["diabetes"], res["smoking"], res["liver_disease"], res["gene_risk"], False, a_risk)
                
                if impacts:
                    st.plotly_chart(plot_drug_disease_impact(drug, impacts), use_container_width=True, key=f"disease_{drug}_{idx}")
                
                st.markdown("**Patient Factor Contributions:**")
                # Fixed positional argument call
                fig = plot_shap_waterfall(s_vals, a_risk, res["risk_percent"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"shap_{drug}_{idx}")
                else:
                    st.info("No specific patient risk factors are impacting this medication.")
    st.markdown('</div>', unsafe_allow_html=True)
