from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from model.predict import predict_adr, drug_encoder

app = FastAPI()


# -------------------------------
# Request Models
# -------------------------------

class Drug(BaseModel):
    name: str
    dose: float


class PatientRequest(BaseModel):
    age: int
    bp: int
    diabetes: bool
    smoking: bool
    liver_disease: bool
    gene_risk: bool
    family_history: bool
    drugs: List[Drug]


# -------------------------------
# Predict Endpoint
# -------------------------------

@app.post("/predict")
def predict(data: PatientRequest):

    drug_ids = []

    # Case-insensitive safe matching
    available_drugs = list(drug_encoder.classes_)
    available_lower = [d.lower().strip() for d in available_drugs]

    for drug in data.drugs:
        drug_name = drug.name.strip().lower()

        if drug_name in available_lower:
            index = available_lower.index(drug_name)
            true_name = available_drugs[index]
            drug_id = drug_encoder.transform([true_name])[0]
            drug_ids.append(drug_id)
        else:
            return {
                "error": f"Drug '{drug.name}' not found",
                "available_drugs_sample": available_drugs[:20]
            }

    # Prepare lab features
    lab_features = [[
        data.age,
        data.bp,
        int(data.diabetes),
        int(data.smoking),
        int(data.liver_disease),
        int(data.gene_risk),
        int(data.family_history),
        0,
        0,
        0
    ]]

    # Model prediction
    result = predict_adr(drug_ids, lab_features)

    risk_percent = round(result * 100, 2)

    # Risk level logic
    if risk_percent < 40:
        level = "Low Risk"
        recommendation = "Safe to Take"
    elif risk_percent < 70:
        level = "Moderate Risk"
        recommendation = "Use With Caution"
    else:
        level = "High Risk"
        recommendation = "Avoid This Medication"

    return {
        "risk_probability": result,
        "risk_percent": risk_percent,
        "risk_level": level,
        "recommendation": recommendation
    }