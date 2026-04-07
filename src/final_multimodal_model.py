import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class FinalADRModel(nn.Module):
    def __init__(self, num_nodes, lab_dim=9, embed_dim=32, hidden_dim=64):
        super().__init__()

        # Drug graph embedding
        self.embedding = nn.Embedding(num_nodes, embed_dim)
        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Patient lab encoder
        self.lab_encoder = nn.Sequential(
            nn.Linear(lab_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, edge_index, drug_ids, lab_features):

        x = self.embedding.weight
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        drug_embed = x[drug_ids]

        lab_embed = self.lab_encoder(lab_features)

        combined = torch.cat([drug_embed, lab_embed], dim=1)

        output = self.fusion(combined)

        return torch.sigmoid(output).squeeze()