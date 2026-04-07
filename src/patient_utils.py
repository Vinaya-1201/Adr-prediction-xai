import torch

def create_patient_vector(age, gender, bp, diabetes,
                          smoking, liver_disease,
                          gene_risk, family_history):

    age_norm = age / 100
    bp_norm = bp / 200

    gender_flag = 1 if gender.lower() == "female" else 0
    diabetes_flag = 1 if diabetes else 0
    smoking_flag = 1 if smoking else 0
    liver_flag = 1 if liver_disease else 0
    gene_flag = 1 if gene_risk else 0
    family_flag = 1 if family_history else 0

    patient_vector = torch.tensor([
        age_norm,
        gender_flag,
        bp_norm,
        diabetes_flag,
        smoking_flag,
        liver_flag,
        gene_flag,
        family_flag
    ]).float()

    return patient_vector