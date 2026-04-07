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
        
        # Ensure tensors are 1D/2D properly
        if lab_tensor.dim() == 1:
            lab_tensor = lab_tensor.unsqueeze(0)
        if drug_tensor.dim() == 0:
            drug_tensor = drug_tensor.unsqueeze(0)

        output = model(edge_index, drug_tensor, lab_tensor)
        # Handle output shape - convert to scalar if needed
        if isinstance(output, torch.Tensor):
            output = output.item() if output.numel() == 1 else output.mean().item()
        return output
drug_encoder = joblib.load(os.path.join(BASE_DIR, "backend/model/drug_encoder.pkl"))