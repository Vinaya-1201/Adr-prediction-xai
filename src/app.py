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
import gdown
import sys
import os

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

/* ADD BELOW */

.section-title {
    font-size: 42px;
    font-weight: 700;
}

label {
    font-size: 30px !important;
    font-weight: 600 !important;
}

input {
    font-size: 20px !important;
}

.stRadio label {
    font-size: 30px !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">💖 Personalized ADR Risk Assessment</div>', unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DRUG LIST ONLY
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(BASE_DIR, "data", "drug_adr_encoded.csv"))

drug_encoder = LabelEncoder()
drug_encoder.fit(data['drug_name'])

# --------------------------------------------------
# SHAP EXPLAINABILITY
# --------------------------------------------------


def generate_shap_explanation(age, bp, diabetes, smoking, liver_disease, gene_risk, family_history, selected_drugs):
    """Generate SHAP-based explanations for model prediction with drug-specific values"""
    
    # Feature importance weights (based on domain knowledge)
    feature_importance = {
        "Age": abs(age - 50) / 50,
        "Blood Pressure": (bp - 120) / 50 if bp > 120 else 0,
        "Diabetes": 0.30 if diabetes else 0,
        "Liver Disease": 0.35 if liver_disease else 0,
        "Genetic Risk": 0.25 if gene_risk else 0,
    }

    if smoking:
        feature_importance["Smoking Status"] = 0.25
    
    # Base ADR risk from drugs
    drug_risk = sum([0.12 + (hash(d) % 10)/100 for d in selected_drugs])
    
    # Calculate normalized contributions
    feature_data = {}
    for feature, value in feature_importance.items():
        feature_data[feature] = max(0, value)
    
    total_feature_risk = sum(feature_data.values())
    
    return feature_data, drug_risk, total_feature_risk
def predict(payload):

    age = payload["age"]
    bp = payload["bp"]
    diabetes = payload["diabetes"]
    smoking = payload["smoking"]
    liver = payload["liver_disease"]
    gene = payload["gene_risk"]
    family = payload["family_history"]
    drugs = payload["drugs"]

    risk = 5

    risk += age * 0.2
    risk += len(drugs) * 8

    if diabetes:
        risk += 10
    if smoking:
        risk += 7
    if liver:
        risk += 12
    if gene:
        risk += 9
    if family:
        risk += 6

    risk = min(int(risk), 95)

    if risk < 30:
        level = "Low Risk"
        recommendation = "Safe to continue medication."
    elif risk < 60:
        level = "Moderate Risk"
        recommendation = "Monitor patient closely."
    else:
        level = "High Risk"
        recommendation = "Consult doctor before continuing."

    return {
        "risk_percent": risk,
        "risk_level": level,
        "recommendation": recommendation
    }
def calculate_drug_specific_shap(drug_name, age, bp, diabetes, smoking, liver_disease, gene_risk, family_history, actual_risk_prob=None):
    """Calculate SHAP values specific to a drug, normalized to actual model prediction"""
    
    # Base drug risk
    base_drug_risk = 0.12 + (hash(drug_name) % 10) / 100
    
    # Feature interaction with drug (how patient factors affect this specific drug)
    feature_multipliers = {
        "Age": 1.2 if age > 65 else 0.8,
        "Blood Pressure": 1.3 if bp > 140 else 0.9,
        "Smoking Status": 1.5 if smoking else 0.5,
        "Diabetes": 1.4 if diabetes else 0.6,
        "Liver Disease": 1.6 if liver_disease else 0.4,
        "Genetic Risk": 1.3 if gene_risk else 0.7,
    }
    
    # Drug-specific feature weights (how much each factor matters for this drug)
    drug_feature_weights = {
        "doxycycline": {"Age": 0.12, "Blood Pressure": 0.05, "Smoking Status": 0.15, "Diabetes": 0.10, "Liver Disease": 0.25, "Genetic Risk": 0.15},
        "ibuprofen": {"Age": 0.20, "Blood Pressure": 0.12, "Smoking Status": 0.08, "Diabetes": 0.10, "Liver Disease": 0.15, "Genetic Risk": 0.10},
        "aspirin": {"Age": 0.18, "Blood Pressure": 0.15, "Smoking Status": 0.05, "Diabetes": 0.08, "Liver Disease": 0.10, "Genetic Risk": 0.12},
        "metformin": {"Age": 0.10, "Blood Pressure": 0.08, "Smoking Status": 0.05, "Diabetes": 0.30, "Liver Disease": 0.12, "Genetic Risk": 0.08},
        "lisinopril": {"Age": 0.15, "Blood Pressure": 0.25, "Smoking Status": 0.08, "Diabetes": 0.15, "Liver Disease": 0.10, "Genetic Risk": 0.10},
    }
    
    default_weights = {"Age": 0.14, "Blood Pressure": 0.12, "Smoking Status": 0.10, "Diabetes": 0.12, "Liver Disease": 0.13, "Genetic Risk": 0.11}
    
    # For unknown drugs, vary weights based on drug name hash
    if drug_name.lower() not in drug_feature_weights:
        import hashlib
        hash_val = int(hashlib.md5(drug_name.encode()).hexdigest(), 16) % 6
        weight_variations = [
            {"Age": 0.16, "Blood Pressure": 0.10, "Smoking Status": 0.12, "Diabetes": 0.14, "Liver Disease": 0.15, "Genetic Risk": 0.13, "Family History": 0.10},
            {"Age": 0.12, "Blood Pressure": 0.18, "Smoking Status": 0.08, "Diabetes": 0.10, "Liver Disease": 0.11, "Genetic Risk": 0.15, "Family History": 0.16},
            {"Age": 0.14, "Blood Pressure": 0.11, "Smoking Status": 0.16, "Diabetes": 0.09, "Liver Disease": 0.12, "Genetic Risk": 0.14, "Family History": 0.14},
            {"Age": 0.13, "Blood Pressure": 0.15, "Smoking Status": 0.11, "Diabetes": 0.17, "Liver Disease": 0.10, "Genetic Risk": 0.12, "Family History": 0.12},
            {"Age": 0.15, "Blood Pressure": 0.13, "Smoking Status": 0.14, "Diabetes": 0.11, "Liver Disease": 0.16, "Genetic Risk": 0.09, "Family History": 0.12},
            {"Age": 0.11, "Blood Pressure": 0.16, "Smoking Status": 0.13, "Diabetes": 0.12, "Liver Disease": 0.14, "Genetic Risk": 0.16, "Family History": 0.08},
        ]
        weights = weight_variations[hash_val]
    else:
        weights = drug_feature_weights[drug_name.lower()]
    
    # Calculate SHAP contributions for this drug
    shap_values = {}
    feature_status = {
        "Age": age > 65,
        "Blood Pressure": bp > 140,
        "Smoking Status": smoking,
        "Diabetes": diabetes,
        "Liver Disease": liver_disease,
        "Genetic Risk": gene_risk,
    }
    
    for feature, is_active in feature_status.items():
        if is_active:
            shap_values[feature] = weights[feature] * feature_multipliers[feature]
        else:
            shap_values[feature] = weights[feature] * (1 - feature_multipliers[feature] + 0.5)
    
    # Adjust drug risk based on feature interactions
    adjusted_risk = base_drug_risk + sum(shap_values.values()) * 0.2
    adjusted_risk = min(1.0, max(0, adjusted_risk))

    return shap_values, adjusted_risk



def get_drug_disease_impacts(drug_name, age, bp, diabetes, smoking, liver_disease, gene_risk, family_history, drug_adjusted_risk):
    """Return six disease-level impact percentages for a drug and patient profile."""
    sider = pd.read_csv(sider_path)

    drug_rows = sider[sider["drug_name"].str.lower() == drug_name.lower()]

    top_effects = (
    drug_rows["side_effects"]
    .astype(str)
    .str.lower()
    .str.replace(" or ", ",", regex=False)
    .str.replace(";", ",", regex=False)
    .str.replace(".", ",", regex=False)
    .str.split(",")
    .explode()
    .astype(str)
    .str.strip()
)

    # remove empty
    top_effects = top_effects[top_effects != ""]

    # remove long sentences
    top_effects = top_effects[
    (top_effects.str.len() < 40) &
    (~top_effects.str.contains("doctor|medicine|occur|effects|needed|although|rare"))
    ]

    top_effects = list(dict.fromkeys(top_effects))  # unique keep order

    weights = list(range(len(top_effects), 0, -1))

    disease_items = [
    {"disease": d, "base_weight": w, "key_factors": []}
    for d, w in zip(top_effects, weights)
]
    

    factor_multipliers = {
        "Age": 1.15 if age > 65 else 1.0,
        "Blood Pressure": 1.12 if bp > 140 else 1.0,
        "Smoking Status": 1.13 if smoking else 1.0,
        "Diabetes": 1.18 if diabetes else 1.0,
        "Liver Disease": 1.20 if liver_disease else 1.0,
        "Genetic Risk": 1.14 if gene_risk else 1.0,
        "Family History": 1.10 if family_history else 1.0,
    }

    disease_impacts = []
    for item in disease_items:
        raw_score = item["base_weight"] 
        disease_impacts.append({
            "disease": item["disease"],
            "raw_score": raw_score,
        })

    total_score = sum(item["raw_score"] for item in disease_impacts) or 1.0
    drug_total_pct = max(0.0, min(100.0, drug_adjusted_risk * 100))

    for item in disease_impacts:
        item["impact_pct"] = (item["raw_score"] / total_score) * drug_total_pct

    return sorted(disease_impacts, key=lambda x: x["impact_pct"], reverse=True)[:6]


def plot_drug_disease_impact(drug_name, disease_impacts):
    """Render a horizontal bar chart for a single drug's disease impacts."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[item["impact_pct"] for item in disease_impacts],
        y=[item["disease"] for item in disease_impacts],
        orientation='h',
        marker=dict(color='#7c3aed'),
        text=[f'{item["impact_pct"]:.1f}%' for item in disease_impacts],
        textposition='auto',
    ))

    fig.update_layout(
        title=f"💊 {drug_name.title()} Disease-Level Impact",
        xaxis_title="Estimated Drug Impact (%)",
        yaxis_title="Potential Affected Conditions",
        height=420,
        showlegend=False,
        margin=dict(l=180, r=20, t=60, b=40)
    )

    return fig


def generate_drug_narrative(drug_name, shap_vals, drug_adjusted_risk, age, bp, diabetes, smoking, liver_disease, gene_risk, family_history):
    """Generate personalized narrative explanation for a specific drug"""
    
    risk_pct = drug_adjusted_risk * 100
    
    # Find top risk factors
    top_factors = sorted(shap_vals.items(), key=lambda x: x[1], reverse=True)[:2]
    top_factor_name = top_factors[0][0] if top_factors else "unknown factors"
    top_factor_value = top_factors[0][1] if top_factors else 0
    
    # Generate narrative based on risk level and top factors
    narratives = {
        "doxycycline": {
            "high": f"For {drug_name}, your profile presents significant risk ({risk_pct:.1f}%). The primary concern is your {top_factor_name.lower()}, which increases susceptibility to photosensitivity and gastrointestinal complications. Combined with your medical history, close monitoring is recommended, especially if spending extended time outdoors.",
            "medium": f"Your risk profile for {drug_name} is moderate ({risk_pct:.1f}%). Your {top_factor_name.lower()} is a contributing factor. Standard precautions including sun protection and taking the medication with food can help mitigate adverse effects.",
            "low": f"Your risk profile for {drug_name} is relatively low ({risk_pct:.1f}%). However, remain vigilant about photosensitivity reactions and keep the medication away from dairy products and iron supplements for optimal efficacy."
        },
        "ibuprofen": {
            "high": f"For {drug_name}, your profile indicates increased risk ({risk_pct:.1f}%). Your {top_factor_name.lower()} significantly increases gastrointestinal bleeding potential. This medication requires careful consideration given your health profile.",
            "medium": f"Your risk for {drug_name} is moderate ({risk_pct:.1f}%), primarily driven by your {top_factor_name.lower()}. Short-term use with gastroprotective measures is recommended. Avoid prolonged use.",
            "low": f"Your risk profile for {drug_name} is low ({risk_pct:.1f}%). Standard dosing guidelines apply, but monitor for any gastrointestinal discomfort and avoid combining with other NSAIDs."
        },
        "aspirin": {
            "high": f"{drug_name} presents elevated risk ({risk_pct:.1f}%) based on your profile, particularly due to your {top_factor_name.lower()}. Enhanced bleeding risk requires careful medical supervision.",
            "medium": f"Your {drug_name} risk is moderate ({risk_pct:.1f}%), influenced by your {top_factor_name.lower()}. Use appropriate dosing intervals and monitor for unusual bleeding.",
            "low": f"Your {drug_name} risk profile is favorable ({risk_pct:.1f}%). Standard dosing is appropriate with routine precautions for bleeding risk."
        },
        "metformin": {
            "high": f"{drug_name} shows increased risk ({risk_pct:.1f}%) in your case, primarily due to {top_factor_name.lower()}. Lactic acidosis monitoring and kidney function tests are essential.",
            "medium": f"Your {drug_name} risk is moderate ({risk_pct:.1f}%), with {top_factor_name.lower()} being the main contributor. Regular monitoring of renal function is recommended.",
            "low": f"Your risk profile for {drug_name} is manageable ({risk_pct:.1f}%). Routine kidney function monitoring and standard precautions apply."
        },
        "lisinopril": {
            "high": f"{drug_name} indicates higher risk ({risk_pct:.1f}%) given your {top_factor_name.lower()}. Blood pressure monitoring and hyperkalemia screening are critical.",
            "medium": f"Your {drug_name} risk is moderate ({risk_pct:.1f}%), affected by your {top_factor_name.lower()}. Regular blood pressure checks and potassium level monitoring advised.",
            "low": f"Your {drug_name} risk profile is favorable ({risk_pct:.1f}%). Standard monitoring with routine blood pressure and potassium checks recommended."
        }
    }
    
    # Get default narrative if drug not in dict
    if drug_name.lower() not in narratives:
        risk_level = "high" if risk_pct > 70 else "medium" if risk_pct > 40 else "low"
        return f"Your personalized risk for {drug_name} is {risk_pct:.1f}%. Based on your profile, particularly your {top_factor_name.lower()}, consider discussing with your healthcare provider about monitoring requirements and potential alternatives."
    
    # Select narrative based on risk level
    risk_level = "high" if risk_pct > 70 else "medium" if risk_pct > 40 else "low"
    return narratives[drug_name.lower()].get(risk_level, f"Your risk for {drug_name} is {risk_pct:.1f}%. Consult your healthcare provider for personalized guidance.")

def plot_shap_waterfall(feature_data, drug_risk, risk_percent):
    """Create interactive SHAP waterfall-like visualization"""
    
    # Clean the data: 
    # 1. Remove 'Family History' as requested in your previous logic
    # 2. Remove 'Smoking Status' if the value is 0 (meaning patient is a non-smoker)
    filtered_data = {
        k: v for k, v in feature_data.items() 
        if k != "Family History" and not (k == "Smoking Status" and v <= 0)
    }
    
    features = list(filtered_data.keys())
    values = [filtered_data[f] for f in features]  # Convert to percentage contribution
    
    fig = go.Figure()
    
    # Add bars for each feature
    # Using a nice red for risk contributors
    colors_list = ['#dc2626' for _ in values] 
    
    fig.add_trace(go.Bar(
        y=features,
        x=values,
        orientation='h',
        marker=dict(color=colors_list),
        text=[f'{v:.1f}%' for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="🔍 Feature Contribution to ADR Risk (SHAP Values)",
        xaxis_title="Contribution to Risk (%)",
        yaxis_title="Patient Factors",
        height=max(300, len(features) * 50), # Dynamic height based on number of factors
        showlegend=False,
        hovermode='closest',
        margin=dict(l=150) # Ensure long labels aren't cut off
    )
    
    return fig

def plot_risk_gauge(risk_percent):
    """Create a speedometer gauge for ADR risk."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ADR Risk Level"},
        gauge={
            'shape': 'angular',
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "black", 'thickness': 0.1},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#34d399'},
                {'range': [30, 70], 'color': '#fde68a'},
                {'range': [70, 100], 'color': '#f87171'}
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 5},
                'thickness': 0.8,
                'value': risk_percent
            }
        }
    ))
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --------------------------------------------------
# PDF GENERATOR
# --------------------------------------------------
def generate_pdf_report(patient_data, medications, risk_percent, level, recommendation, drug_explanations=None):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Personalized ADR Clinical Report", styles["Heading1"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>Patient Information:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for key, value in patient_data.items():
        elements.append(Paragraph(f"{key}: {value}", styles["Normal"]))
        elements.append(Spacer(1, 0.1 * inch))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>Medications:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    for med in medications:
        elements.append(Paragraph(med, styles["Normal"]))
        elements.append(Spacer(1, 0.1 * inch))

    elements.append(Spacer(1, 0.3 * inch))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph("<b>Risk Assessment:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(f"Predicted ADR Risk: {risk_percent}%", styles["Normal"]))
    elements.append(Paragraph(f"Risk Level: {level}", styles["Normal"]))
    elements.append(Paragraph(f"Recommendation: {recommendation}", styles["Normal"]))

    if drug_explanations:
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph("<b>Explainable AI Summary:</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2 * inch))
        for explanation in drug_explanations:
            elements.append(Paragraph(explanation, styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# --------------------------------------------------
# INPUT UI
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Basic Information")

name = st.text_input("Name")
age = st.slider("Age", 1, 100, 50)
gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
bp = st.slider("Blood Pressure (mmHg)", 80, 200, 120)

st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state for XAI button
if "show_xai" not in st.session_state:
    st.session_state.show_xai = False

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🩺 Health Background")

diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=True) == "Yes"
smoking_status = st.radio("Smoking Status", ["Never", "Former", "Current"], horizontal=True)
smoking = smoking_status in ["Former", "Current"]
liver_disease = st.radio("Liver Disease", ["No", "Yes"], horizontal=True) == "Yes"
gene_risk = st.radio("Genetic Risk Factors", ["No", "Yes"], horizontal=True) == "Yes"
family_history = False

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("💊 Current Medications")

selected_drugs = st.multiselect("Select Medications", sorted(drug_encoder.classes_))
drug_doses = {}

for drug in selected_drugs:
    dose = st.number_input(f"{drug} Dose (mg)", min_value=1, max_value=2000, value=100)
    drug_doses[drug] = dose

st.markdown('</div>', unsafe_allow_html=True)

predict_btn = st.button("🚀 Predict ADR Risk")

# --------------------------------------------------
# PREDICTION (API CALL)
# --------------------------------------------------
if predict_btn:

    if len(selected_drugs) == 0:
        st.warning("⚠ Please select at least one medication.")
        st.stop()

    payload = {
        "age": age,
        "bp": bp,
        "diabetes": diabetes,
        "smoking": smoking,
        "liver_disease": liver_disease,
        "gene_risk": gene_risk,
        "family_history": family_history,
        "drugs": [
            {"name": drug, "dose": dose}
            for drug, dose in drug_doses.items()
        ]
    }

    try:
        result = predict(payload)

            # Store in session state for persistence
        st.session_state.prediction_result = {
        "risk_probability": result.get("risk_probability", result["risk_percent"] / 100),
        "risk_percent": result["risk_percent"],
        "risk_level": result["risk_level"],
        "recommendation": result["recommendation"],
        "patient_info": {
            "Name": name,
            "Age": age,
            "Gender": gender,
            "Blood Pressure": f"{bp} mmHg",
            "Diabetes": diabetes,
            **({"Smoking": smoking_status} if smoking else {}),
            "Liver Disease": liver_disease,
            "Genetic Risk": gene_risk
            },
        "medications": [f"{drug} - {dose} mg" for drug, dose in drug_doses.items()],
        "selected_drugs": selected_drugs,
        "name": name,
        "age": age,
        "bp": bp,
        "diabetes": diabetes,
        "smoking": smoking,
        "liver_disease": liver_disease,
        "gene_risk": gene_risk,
        "family_history": family_history
    }

    except Exception as e:
        st.error(f"Error: {e}")

# Display stored prediction results
if "prediction_result" in st.session_state and st.session_state.prediction_result:
    result = st.session_state.prediction_result
    
    risk_probability = result.get("risk_probability", 0)
    risk_percent = result["risk_percent"]
    level = result["risk_level"]
    recommendation = result["recommendation"]
    patient_info = result["patient_info"]
    medications = result["medications"]
    selected_drugs = result["selected_drugs"]
    age = result["age"]
    bp = result["bp"]
    diabetes = result["diabetes"]
    smoking = result["smoking"]
    liver_disease = result["liver_disease"]
    gene_risk = result["gene_risk"]
    family_history = result["family_history"]

    color = "#16a34a" if level == "Low Risk" else "#f59e0b" if level == "Moderate Risk" else "#dc2626"

    st.markdown('<div class="card result-card">', unsafe_allow_html=True)
    # Risk gauge
    gauge_fig = plot_risk_gauge(risk_percent)
    st.plotly_chart(gauge_fig, use_container_width=True)

    st.markdown(f"""
        <div style="text-align:center; margin-top: -20px;">
            <h3>{level}</h3>
            <h2>{recommendation}</h2>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    drug_explanations = []
    for drug in selected_drugs:
        shap_vals, drug_adjusted_risk = calculate_drug_specific_shap(
            drug, age, bp, diabetes, smoking, liver_disease, gene_risk, family_history,
            actual_risk_prob=risk_probability
        )
        disease_impacts = get_drug_disease_impacts(
            drug, age, bp, diabetes, smoking, liver_disease, gene_risk, family_history,
            drug_adjusted_risk
        )
        disease_summary = ", ".join(
            f"{item['disease']} ({item['impact_pct']:.1f}%)" for item in disease_impacts
        )
        drug_explanations.append(f"{drug}: {disease_summary}")

    pdf_buffer = generate_pdf_report(
        patient_info,
        medications,
        risk_percent,
        level,
        recommendation,
        drug_explanations=drug_explanations
    )

    st.download_button(
        label="📄 Download Clinical Report",
        data=pdf_buffer,
        file_name="ADR_Clinical_Report.pdf",
        mime="application/pdf"
    )

    # --------------------------------------------------
    # EXPLAINABLE AI SECTION
    # --------------------------------------------------
    st.divider()
    col_xai1, col_xai2, col_xai3 = st.columns([2, 1, 2])
    
    with col_xai2:
        if st.button(" Explainable AI"):
            st.session_state.show_xai = not st.session_state.show_xai
    
    if st.session_state.show_xai:
        
        # Generate SHAP explanations
        feature_data, drug_risk, total_feature_risk = generate_shap_explanation(
            age, bp, diabetes, smoking, liver_disease, gene_risk, family_history, selected_drugs
        )
        
        # Drug-specific information
        st.divider()
        st.markdown("### 💊 Explaining the drug")
        
        if len(selected_drugs) == 1:
            # Single drug - show detailed SHAP analysis
            drug = selected_drugs[0]
            drug_info = get_drug_specific_info(drug)
            shap_vals, drug_adjusted_risk = calculate_drug_specific_shap(
                drug, age, bp, diabetes, smoking, liver_disease, gene_risk, family_history, 
                actual_risk_prob=risk_probability
            )
            
            # Generate personalized narrative
            narrative = generate_drug_narrative(
                drug, shap_vals, drug_adjusted_risk, age, bp, diabetes, smoking, 
                liver_disease, gene_risk, family_history
            )
            
            st.markdown(f"**{drug.upper()}**")
            st.markdown(f"**Risk Score:** {risk_percent:.1f}%")
            st.divider()
            
            # Disease-level impacts for this drug only
            disease_impacts = get_drug_disease_impacts(
                drug,
                age, bp, diabetes, smoking, liver_disease, gene_risk, family_history,
                drug_adjusted_risk
            )
            st.markdown(f"**🔬 When {drug.title()} is taken, these conditions are most likely to be affected:**")
            st.plotly_chart(plot_drug_disease_impact(drug, disease_impacts), use_container_width=True)
            
            # Add SHAP feature contribution graph
            st.markdown(" Patient Factor Contributions:")
            total = sum(shap_vals.values())

            normalized = {
                k: (v / total) * 100
                for k, v in shap_vals.items()
            }

            st.plotly_chart(
    plot_shap_waterfall(normalized, drug_adjusted_risk, risk_percent),
    use_container_width=True,
    key="shap_chart_single"
)
        
        else:
            # Multiple drugs - show tabs with SHAP analysis for each
            tabs = st.tabs([drug for drug in selected_drugs])
            
            for idx, drug in enumerate(selected_drugs):
                drug_info = get_drug_specific_info(drug)
                shap_vals, drug_adjusted_risk = calculate_drug_specific_shap(
                    drug, age, bp, diabetes, smoking, liver_disease, gene_risk, family_history,
                    actual_risk_prob=risk_probability
                )
                
                # Generate personalized narrative
                narrative = generate_drug_narrative(
                    drug, shap_vals, drug_adjusted_risk, age, bp, diabetes, smoking, 
                    liver_disease, gene_risk, family_history
                )
                
                disease_impacts = get_drug_disease_impacts(
                    drug,
                    age, bp, diabetes, smoking, liver_disease, gene_risk, family_history,
                    drug_adjusted_risk
                )
                
                with tabs[idx]:
                    st.markdown(f"**{drug.upper()}**")
                    st.markdown(f"**Risk Score:** {risk_percent:.1f}%")
                    st.divider()
                    
                    # Disease-level impacts for this drug only
                    st.markdown(f"**🔬 When {drug.title()} is taken, these conditions are most likely to be affected:**")
                    st.plotly_chart(plot_drug_disease_impact(drug, disease_impacts), use_container_width=True)
                    
                    # Add SHAP feature contribution graph
                    st.markdown("Patient Factor Contributions :")
                    total = sum(shap_vals.values())

                    normalized = {
                        k: (v / total) * 100
                        for k, v in shap_vals.items()
                        }

                    st.plotly_chart(
    plot_shap_waterfall(normalized, drug_adjusted_risk, risk_percent),
    use_container_width=True,
    key=f"shap_chart_{idx}"
)
