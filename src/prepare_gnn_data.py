import pandas as pd
import torch
from torch_geometric.data import Data
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load combined dataset
data = pd.read_csv(os.path.join(BASE_DIR, "data", "combined_drug_adr.csv"))

# Count nodes
num_drugs = data['drug_id'].nunique()
num_adrs = data['adr_id'].nunique()

edge_list = []

for _, row in data.iterrows():
    drug_node = row['drug_id']
    adr_node = row['adr_id'] + num_drugs   # SHIFT ADR IDS

    edge_list.append([drug_node, adr_node])
    edge_list.append([adr_node, drug_node])  # undirected graph

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

num_nodes = num_drugs + num_adrs

# Node features
x = torch.randn((num_nodes, 32))  # increased dimension

y = torch.zeros(num_nodes, dtype=torch.long)

graph_data = Data(x=x, edge_index=edge_index, y=y)

print(graph_data)