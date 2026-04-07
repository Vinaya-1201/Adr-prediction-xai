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

    risk += age * 0.15
    risk += len(drugs) * 10

    if diabetes:
        risk += 8
    if smoking:
        risk += 6
    if liver:
        risk += 12
    if gene:
        risk += 10
    if family:
        risk += 5

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
