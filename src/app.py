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
import os
import gdown

os.makedirs("data", exist_ok=True)

if not os.path.exists("data/sider.csv"):
    url = "https://drive.google.com/uc?id=1NjuGqaKElyeY-ovqTyEfr3xtXY-w7rY8"
    gdown.download(url, "data/sider.csv", quiet=False)
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
# LOAD DRUG LIST ONLY
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(BASE_DIR, "data", "drug_adr_encoded.csv"))

drug_encoder = LabelEncoder()
drug_encoder.fit(data['drug_name'])

# --------------------------------------------------
# PDF GENERATOR
# --------------------------------------------------
def generate_pdf_report(patient_data, medications, risk_percent, level, recommendation):

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

    doc.build(elements)
    buffer.seek(0)
    return buffer

# --------------------------------------------------
# INPUT UI
# --------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Basic Information")

age = st.slider("Age", 1, 100, 50)
gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
bp = st.slider("Blood Pressure (mmHg)", 80, 200, 120)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🩺 Health Background")

diabetes = st.radio("Diabetes", ["No", "Yes"], horizontal=True) == "Yes"
smoking_status = st.radio("Smoking Status", ["Never", "Former", "Current"], horizontal=True)
smoking = smoking_status in ["Former", "Current"]
liver_disease = st.radio("Liver Disease", ["No", "Yes"], horizontal=True) == "Yes"
gene_risk = st.radio("Genetic Risk Factors", ["No", "Yes"], horizontal=True) == "Yes"
family_history = st.radio("Family History of Drug Reaction", ["No", "Yes"], horizontal=True) == "Yes"

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("💊 Current Medications")

selected_drugs = st.multiselect("Select Medications", sorted(drug_encoder.classes_))
drug_doses = {}

for drug in selected_drugs:
    dose = st.number_input(f"{drug} Dose (mg)", min_value=1, max_value=2000, value=100)
    drug_doses[drug] = dose

st.markdown('</div>', unsafe_allow_html=True)

predict = st.button("🚀 Predict ADR Risk")
# --------------------------------------------------
# PREDICTION (API CALL)
# --------------------------------------------------
if predict:

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
        response = requests.post(
            "http://localhost:8001/predict",
            json=payload
        )

        st.write(response.status_code)
        st.write(response.text)

        if response.status_code == 200:

            result = response.json()

            risk_percent = result["risk_percent"]
            level = result["risk_level"]
            recommendation = result["recommendation"]

            color = "#16a34a" if level == "Low Risk" else "#f59e0b" if level == "Moderate Risk" else "#dc2626"

            st.markdown(f"""
            <div class="card result-card">
                <h1 style="font-size:70px; color:{color};">{risk_percent}%</h1>
                <h3>{level}</h3>
                <h2>{recommendation}</h2>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error("Backend error. Check FastAPI server.")

    except Exception as e:
        st.error(f"Error: {e}")
