import streamlit as st
import pandas as pd
import os
import io
import hashlib
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import os
import gdown

# create data folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(BASE_DIR, "data")
os.makedirs(data_dir, exist_ok=True)

# sider path
sider_path = os.path.join(data_dir, "sider.csv")

# download if missing
if not os.path.exists(sider_path):
    url = "https://drive.google.com/uc?id=1NjuGqaKElyeY-ovqTyEfr3xtXY-w7rY8"
    gdown.download(url, sider_path, quiet=False)
# --------------------------------------------------
# PAGE CONFIG & STYLING
# --------------------------------------------------
st.set_page_config(page_title="Personalized ADR Risk Assessment", page_icon="💖", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #ffe4ec, #fff1f6); font-family: 'Segoe UI', sans-serif; }
.main-title { font-size: 50px; font-weight: 800; color: #9d174d; text-align: center; margin-bottom: 40px; }
.card { background: white; padding: 35px; border-radius: 20px; box-shadow: 0px 10px 30px rgba(0,0,0,0.05); margin-bottom: 35px; }
.result-card { text-align: center; padding: 40px; border-radius: 20px; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💖 Personalized ADR Risk Assessment</div>', unsafe_allow_html=True)

# --------------------------------------------------
# DATA & PROFILES
# --------------------------------------------------
# Assuming drug_adr_encoded.csv is in the data folder
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "drug_adr_encoded.csv"))
    drug_list = sorted(data['drug_name'].unique())
except:
    drug_list = ["doxycycline", "ibuprofen", "aspirin", "metformin", "lisinopril"]

DRUG_PROFILES = {
    "doxycycline": {"risk_level": 0.25, "impacts": [{"disease": "Photosensitivity", "weight": 0.18, "factors": ["Age", "Liver Disease"]}, {"disease": "GI upset", "weight": 0.16, "factors": ["Smoking Status"]}]},
    "ibuprofen": {"risk_level": 0.20, "impacts": [{"disease": "GI bleeding", "weight": 0.20, "factors": ["Age", "Diabetes"]}, {"disease": "Ulcers", "weight": 0.16, "factors": ["Smoking Status"]}]},
    "aspirin": {"risk_level": 0.15, "impacts": [{"disease": "Bleeding", "weight": 0.22, "factors": ["Age"]}, {"disease": "GI upset", "weight": 0.16, "factors": ["Smoking Status"]}]}
}

# --------------------------------------------------
# CORE LOGIC FUNCTIONS
# --------------------------------------------------

def calculate_drug_specific_shap(drug_name, age, bp, diabetes, smoking, liver_disease, gene_risk, family_history):
    base_drug_risk = DRUG_PROFILES.get(drug_name.lower(), {}).get("risk_level", 0.15)
    
    # Feature multipliers: Force 0 if condition is not met
    multipliers = {
        "Age": 1.2 if age > 65 else 0.8,
        "Blood Pressure": 1.3 if bp > 140 else 0.9,
        "Smoking Status": 1.5 if smoking else 0.0, # FIXED: Strict 0
        "Diabetes": 1.4 if diabetes else 0.6,
        "Liver Disease": 1.6 if liver_disease else 0.4,
        "Genetic Risk": 1.3 if gene_risk else 0.7,
    }
    
    weights = {"Age": 0.15, "Blood Pressure": 0.10, "Smoking Status": 0.12, "Diabetes": 0.10, "Liver Disease": 0.15, "Genetic Risk": 0.12}
    
    shap_values = {}
    active_map = {"Age": age > 65, "Blood Pressure": bp > 140, "Smoking Status": smoking, "Diabetes": diabetes, "Liver Disease": liver_disease, "Genetic Risk": gene_risk}
    
    for feature, is_active in active_map.items():
        shap_values[feature] = (weights[feature] * multipliers[feature]) if is_active else 0.0
            
    adjusted_risk = base_drug_risk + (sum(shap_values.values()) * 0.2)
    return shap_values, min(1.0, max(0, adjusted_risk))

def plot_shap_waterfall(feature_data, drug_risk, risk_percent):
    """Filters out zero-value bars and Family History"""
    filtered = {k: v for k, v in feature_data.items() if v > 0 and k != "Family History"}
    if not filtered: return None

    fig = go.Figure(go.Bar(
        y=list(filtered.keys()), x=[v * 100 for v in filtered.values()],
        orientation='h', marker=dict(color='#dc2626'),
        text=[f'{v*100:.1f}%' for v in filtered.values()], textposition='auto'
    ))
    fig.update_layout(title="🔍 Patient Factor Contributions", xaxis_title="Contribution (%)", height=300, margin=dict(l=150))
    return fig

def plot_risk_gauge(risk_percent):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=risk_percent, gauge={'axis': {'range': [0, 100]}, 'steps': [{'range': [0, 35], 'color': "green"}, {'range': [70, 100], 'color': "red"}]}))
    fig.update_layout(height=250)
    return fig

# --------------------------------------------------
# MAIN UI
# --------------------------------------------------
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
selected_drugs = st.multiselect("Select Medications", drug_list)
st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 Predict ADR Risk"):
    if not selected_drugs:
        st.warning("Please select at least one drug.")
    else:
        risk_val = 20.0 + (15.0 if smoking else 0) + (len(selected_drugs) * 5)
        st.session_state.prediction_result = {
            "risk_percent": min(risk_val, 100.0), "level": "High" if risk_val > 60 else "Moderate" if risk_val > 30 else "Low",
            "age": age, "bp": bp, "diabetes": diabetes, "smoking": smoking, "liver_disease": liver_disease, "gene_risk": gene_risk, "selected_drugs": selected_drugs
        }

if "prediction_result" in st.session_state:
    res = st.session_state.prediction_result
    st.markdown('<div class="card result-card">', unsafe_allow_html=True)
    st.plotly_chart(plot_risk_gauge(res["risk_percent"]), use_container_width=True)
    st.markdown(f"### Result: {res['level']} Risk")
    
    if st.button("Explainable AI"):
        st.session_state.show_xai = not st.session_state.get("show_xai", False)
    
    if st.session_state.get("show_xai"):
        tabs = st.tabs(res["selected_drugs"])
        for idx, drug in enumerate(res["selected_drugs"]):
            with tabs[idx]:
                s_vals, a_risk = calculate_drug_specific_shap(drug, res["age"], res["bp"], res["diabetes"], res["smoking"], res["liver_disease"], res["gene_risk"], False)
                st.markdown("**Patient Factor Contributions:**")
                fig = plot_shap_waterfall(s_vals, a_risk, res["risk_percent"])
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"shap_{drug}_{idx}")
                else:
                    st.info("No specific patient risk factors are impacting this medication.")
    st.markdown('</div>', unsafe_allow_html=True)
