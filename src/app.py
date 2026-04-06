import streamlit as st
import pandas as pd
import os
import io
import sys
import gdown
from sklearn.preprocessing import LabelEncoder

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# -----------------------------
# FIX IMPORT PATH
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(ROOT_DIR)

from backend.predict import predict

# -----------------------------
# DOWNLOAD SIDER DATA
# -----------------------------
data_dir = os.path.join(ROOT_DIR, "data")
os.makedirs(data_dir, exist_ok=True)

sider_path = os.path.join(data_dir, "sider.csv")

if not os.path.exists(sider_path):
    gdown.download(
        "https://drive.google.com/uc?id=1NjuGqaKElyeY-ovqTyEfr3xtXY-w7rY8",
        sider_path,
        quiet=False
    )

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="ADR Risk Prediction",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Personalized ADR Risk Prediction")

# -----------------------------
# LOAD DATA
# -----------------------------
data = pd.read_csv(os.path.join(data_dir, "drug_adr_encoded.csv"))

drug_encoder = LabelEncoder()
drug_encoder.fit(data["drug_name"])

# -----------------------------
# PDF REPORT
# -----------------------------
def create_pdf(patient, risk, level):

    buffer = io.BytesIO()
    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    elements.append(Paragraph("ADR Risk Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    for k, v in patient.items():
        elements.append(Paragraph(f"{k}: {v}", styles["Normal"]))
        elements.append(Spacer(1, 6))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Risk: {risk}%", styles["Heading2"]))
    elements.append(Paragraph(level, styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# -----------------------------
# UI INPUT
# -----------------------------
st.subheader("Patient Information")

age = st.slider("Age", 1, 100, 50)
bp = st.slider("Blood Pressure", 80, 200, 120)

diabetes = st.checkbox("Diabetes")
smoking = st.checkbox("Smoking")
liver = st.checkbox("Liver Disease")
genetic = st.checkbox("Genetic Risk")
family = st.checkbox("Family History")

st.subheader("Select Drugs")

selected_drugs = st.multiselect(
    "Drugs",
    sorted(drug_encoder.classes_)
)

dose_dict = {}

for drug in selected_drugs:
    dose_dict[drug] = st.number_input(
        f"{drug} dose",
        1,
        2000,
        100
    )

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🚀 Predict ADR Risk"):

    if len(selected_drugs) == 0:
        st.warning("Select at least one drug")
        st.stop()

    payload = {
        "age": age,
        "bp": bp,
        "diabetes": diabetes,
        "smoking": smoking,
        "liver_disease": liver,
        "gene_risk": genetic,
        "family_history": family,
        "drugs": [
            {"name": d, "dose": dose_dict[d]}
            for d in selected_drugs
        ]
    }

    try:
        result = predict(payload)

        risk = result["risk_percent"]
        level = result["risk_level"]

        st.success(f"ADR Risk: {risk}%")
        st.info(level)

        patient = {
            "Age": age,
            "BP": bp,
            "Drugs": ", ".join(selected_drugs)
        }

        pdf = create_pdf(patient, risk, level)

        st.download_button(
            "Download Report",
            pdf,
            file_name="ADR_Report.pdf"
        )

    except Exception as e:
        st.error(str(e))
