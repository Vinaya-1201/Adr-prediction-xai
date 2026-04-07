import torch
import pandas as pd
import os
from link_prediction_model import LinkPredictor
from prepare_gnn_data import graph_data
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = pd.read_csv(os.path.join(BASE_DIR, "data", "drug_adr_encoded.csv"))

drug_encoder = LabelEncoder()
adr_encoder = LabelEncoder()

drug_encoder.fit(data['drug_name'])
adr_encoder.fit(data['side_effects'])

num_nodes = graph_data.x.shape[0]
model = LinkPredictor(num_nodes=num_nodes)
model.load_state_dict(torch.load("model.pth"))
model.eval()

edge_index = graph_data.edge_index


def predict_top_adrs(drug_name, top_k=5):

    if drug_name not in drug_encoder.classes_:
        print("Drug not found.")
        return

    drug_id = drug_encoder.transform([drug_name])[0]

    with torch.no_grad():
        x = model.embedding.weight
        x = model.conv1(x, edge_index)
        x = torch.relu(x)
        x = model.conv2(x, edge_index)

        drug_embedding = x[drug_id]

        num_adrs = len(adr_encoder.classes_)
        adr_embeddings = x[:num_adrs]

        scores = torch.matmul(adr_embeddings, drug_embedding)
        probs = torch.sigmoid(scores)

        top_probs, top_indices = torch.topk(probs, top_k)

    print(f"\nTop {top_k} predicted ADRs for {drug_name}:\n")

    for idx, prob in zip(top_indices, top_probs):
        adr_name = adr_encoder.inverse_transform([idx.item()])[0]
        print(f"{adr_name} → {round(prob.item()*100, 2)}%")


def explain_drug_similarity(drug_name, top_k=3):

    if drug_name not in drug_encoder.classes_:
        print("Drug not found.")
        return

    drug_id = drug_encoder.transform([drug_name])[0]

    with torch.no_grad():
        x = model.embedding.weight
        x = model.conv1(x, edge_index)
        x = torch.relu(x)
        x = model.conv2(x, edge_index)

        drug_embedding = x[drug_id]

        num_drugs = len(drug_encoder.classes_)
        all_drug_embeddings = x[:num_drugs]

        similarities = torch.matmul(all_drug_embeddings, drug_embedding)
        similarities[drug_id] = -1

        top_sim, top_idx = torch.topk(similarities, top_k)

    print(f"\nDrugs similar to {drug_name}:\n")

    for idx, score in zip(top_idx, top_sim):
        similar_drug = drug_encoder.inverse_transform([idx.item()])[0]
        print(f"{similar_drug} → similarity {round(score.item(),2)}")


if __name__ == "__main__":
    predict_top_adrs("doxycycline", top_k=5)
    explain_drug_similarity("doxycycline", top_k=3)