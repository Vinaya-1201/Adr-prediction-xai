import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class PersonalizedADRModel(nn.Module):
    def __init__(self, num_nodes, embed_dim=32, hidden_dim=64, patient_dim=8):
        super().__init__()

        # Graph Embedding
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Fusion layer (Graph + Patient features)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + patient_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, edge_index, edge_pairs, patient_features):

        # ---- Graph Encoding ----
        x = self.embedding.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        drug_embed = x[edge_pairs[:, 0]]
        adr_embed = x[edge_pairs[:, 1]]

        # Base interaction score
        interaction = drug_embed * adr_embed

        # Concatenate patient features
        combined = torch.cat([interaction, patient_features], dim=1)

        # Final prediction
        output = self.fusion(combined)

        return torch.sigmoid(output).squeeze()