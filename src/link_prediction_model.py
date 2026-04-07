import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class LinkPredictor(torch.nn.Module):
    def __init__(self, num_nodes, embed_dim=32, hidden_dim=64):
        super().__init__()

        # Learnable node embeddings
        self.embedding = torch.nn.Embedding(num_nodes, embed_dim)

        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, edge_index, edge_pairs):
        # Initial embeddings
        x = self.embedding.weight

        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)

        # Get embeddings for drug and ADR nodes
        drug_embed = x[edge_pairs[:, 0]]
        adr_embed = x[edge_pairs[:, 1]]

        # Dot product scoring
        score = (drug_embed * adr_embed).sum(dim=1)

        return torch.sigmoid(score)