import streamlit as st
import pandas as pd
import os
import io
import sys
import gdown
from sklearn.preprocessing import LabelEncoder
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch

# --------------------------------------------------
# FIX BACKEND IMPORT PATH
# --------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from backend.predict import predict

# --------------------------------------------------
# DOWNLOAD SIDER IF NOT EXISTS
# --------------------------------------------------
data_dir = os.path.join(ROOT_DIR, "data")
os.makedirs(data_dir, exist_ok=True)

sider_path = os.path.join(data_dir, "sider.csv")

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

st.title("💖 Personalized ADR Risk Assessment")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
data = pd.read_csv(os.path.join(data_dir, "drug_adr_encoded.csv"))

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
    elements.append(Spacer(1, 12))

    for key, value in patient_data.items():
        elements.append(Paragraph(f"{key}: {value}", styles["Normal"]))
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Risk: {risk_percent}%", styles["Heading2"]))
    elements.append(Paragraph(level, styles["Normal"]))
    elements.append(Paragraph(recommendation, styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# --------------------------------------------------
# INPUT UI
# --------------------------------------------------
st.subheader("👤 Basic Information")

age = st.slider("Age", 1, 100, 50)
gender = st.radio("Gender", ["Male", "Female"])
bp = st.slider("Blood Pressure", 80, 200, 120)

st.subheader("🩺 Health Background")

diabetes = st.checkbox("Diabetes")
smoking = st.checkbox("Smoking")
liver_disease = st.checkbox("Liver Disease")
gene_risk = st.checkbox("Genetic Risk")
family_history = st.checkbox("Family History")

st.subheader("💊 Medications")

selected_drugs = st.multiselect(
    "Select drugs",
    sorted(drug_encoder.classes_)
)

drug_doses = {}

for drug in selected_drugs:
    dose = st.number_input(f"{drug} dose", 1, 2000, 100)
    drug_doses[drug] = dose


predict_btn = st.button("🚀 Predict ADR Risk")

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if predict_btn:

    if len(selected_drugs) == 0:
        st.warning("Select at least one drug")
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
            {"name": d, "dose": drug_doses[d]}
            for d in selected_drugs
        ]
    }

    try:
        result = predict(payload)

        risk_percent = result["risk_percent"]
        level = result["risk_level"]
        recommendation = result["recommendation"]

        st.success(f"{risk_percent}% - {level}")
        st.write(recommendation)

        patient_info = {
            "Age": age,
            "Gender": gender,
            "BP": bp
        }

        meds = [
            f"{d} - {drug_doses[d]} mg"
            for d in selected_drugs
        ]

        pdf = generate_pdf_report(
            patient_info,
            meds,
            risk_percent,
            level,
            recommendation
        )

        st.download_button(
            "Download Report",
            pdf,
            file_name="ADR_Report.pdf"
        )

    except Exception as e:
        st.error(str(e))
