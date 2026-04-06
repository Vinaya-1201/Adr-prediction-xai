import torch
import pandas as pd
import os
from personalized_model import PersonalizedADRModel
from prepare_gnn_data import graph_data
from sklearn.preprocessing import LabelEncoder
from patient_utils import create_patient_vector

# -----------------------------
# Load Data
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(BASE_DIR, "data", "drug_adr_encoded.csv"))

drug_encoder = LabelEncoder()
adr_encoder = LabelEncoder()

drug_encoder.fit(data['drug_name'])
adr_encoder.fit(data['side_effects'])

# -----------------------------
# Load Model
# -----------------------------
num_nodes = graph_data.x.shape[0]

model = PersonalizedADRModel(num_nodes=num_nodes)
model.load_state_dict(torch.load("model.pth"), strict=False)
model.eval()

edge_index = graph_data.edge_index


# =========================================================
# Personalized Prediction Function
# =========================================================
def predict_personalized(drug_name,
                         age, gender, bp,
                         diabetes, smoking,
                         liver_disease,
                         gene_risk,
                         family_history,
                         top_k=5):

    if drug_name not in drug_encoder.classes_:
        print("Drug not found.")
        return

    drug_id = drug_encoder.transform([drug_name])[0]

    patient_vec = create_patient_vector(
        age, gender, bp,
        diabetes, smoking,
        liver_disease,
        gene_risk,
        family_history
    )

    with torch.no_grad():

        # Graph forward pass
        x = model.embedding.weight
        x = model.conv1(x, edge_index)
        x = torch.relu(x)
        x = model.conv2(x, edge_index)

        drug_embedding = x[drug_id]
        adr_embeddings = x[:len(adr_encoder.classes_)]

        interaction = adr_embeddings * drug_embedding

        patient_batch = patient_vec.repeat(len(adr_embeddings), 1)

        combined = torch.cat([interaction, patient_batch], dim=1)

        scores = model.fusion(combined)
        probs = torch.sigmoid(scores).squeeze()

        top_probs, top_indices = torch.topk(probs, top_k)

    print(f"\nPersonalized ADR risk for {drug_name}:\n")

    for idx, prob in zip(top_indices, top_probs):
        adr_name = adr_encoder.inverse_transform([idx.item()])[0]
        print(f"{adr_name} → {round(prob.item()*100, 2)}%")


# =========================================================
# Explainability Function (Gradient-based)
# =========================================================
def explain_prediction(drug_name,
                       age, gender, bp,
                       diabetes, smoking,
                       liver_disease,
                       gene_risk,
                       family_history):

    if drug_name not in drug_encoder.classes_:
        print("Drug not found.")
        return

    drug_id = drug_encoder.transform([drug_name])[0]

    # Create patient vector with gradient tracking
    patient_vec = create_patient_vector(
        age, gender, bp,
        diabetes, smoking,
        liver_disease,
        gene_risk,
        family_history
    )

    patient_vec.requires_grad = True

    # Forward pass (manual graph pass)
    x = model.embedding.weight
    x = model.conv1(x, edge_index)
    x = torch.relu(x)
    x = model.conv2(x, edge_index)

    drug_embedding = x[drug_id]

    # Use first ADR for explanation example
    adr_embedding = x[0]

    interaction = adr_embedding * drug_embedding

    combined = torch.cat([interaction, patient_vec], dim=0).unsqueeze(0)

    score = model.fusion(combined)
    prob = torch.sigmoid(score)

    # Backprop
    prob.backward()

    importance = patient_vec.grad.detach().numpy()

    feature_names = [
        "AGE",
        "GENDER",
        "BP",
        "DIABETES",
        "SMOKING",
        "LIVER_DISEASE",
        "GENE_RISK",
        "FAMILY_HISTORY"
    ]

    print("\nPatient Feature Importance:\n")

    for name, value in zip(feature_names, importance):
        print(f"{name}: {abs(value):.4f}")


# =========================================================
# Run Example
# =========================================================
if __name__ == "__main__":

    predict_personalized(
        drug_name="doxycycline",
        age=65,
        gender="female",
        bp=145,
        diabetes=True,
        smoking=False,
        liver_disease=False,
        gene_risk=True,
        family_history=True
    )

    explain_prediction(
        drug_name="doxycycline",
        age=65,
        gender="female",
        bp=145,
        diabetes=True,
        smoking=False,
        liver_disease=False,
        gene_risk=True,
        family_history=True
    )