import torch
import os
import joblib
from .final_multimodal_model import FinalADRModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

edge_index = torch.load(os.path.join(BASE_DIR, "backend/edge_index.pt"))
num_nodes = torch.load(os.path.join(BASE_DIR, "backend/num_nodes.pt"))

model = FinalADRModel(num_nodes=num_nodes, lab_dim=10)
model.load_state_dict(
    torch.load(os.path.join(BASE_DIR, "backend/model/final_multimodal_model.pth"), map_location="cpu")
)
model.eval()


def predict_adr(drug_ids, lab_features):
    with torch.no_grad():
        lab_tensor = torch.tensor(lab_features, dtype=torch.float)
        drug_tensor = torch.tensor(drug_ids, dtype=torch.long)

        output = model(edge_index, drug_tensor, lab_tensor)
        return output.tolist()
drug_encoder = joblib.load(os.path.join(BASE_DIR, "backend/model/drug_encoder.pkl"))