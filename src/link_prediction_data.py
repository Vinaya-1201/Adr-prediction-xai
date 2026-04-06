import pandas as pd
import torch
import random

# Load encoded data
data = pd.read_csv("data/combined_drug_adr.csv")
# Reduce dataset size for faster training
data = data.sample(50000, random_state=42)

num_drugs = data['drug_id'].nunique()
num_adrs = data['adr_id'].nunique()

# Positive samples
positive_edges = set(zip(data['drug_id'], data['adr_id']))

# Convert to list
positive_edges = list(positive_edges)

# Generate negative samples
negative_edges = set()

while len(negative_edges) < len(positive_edges):
    d = random.randint(0, num_drugs - 1)
    a = random.randint(0, num_adrs - 1)

    if (d, a) not in positive_edges:
        negative_edges.add((d, a))

negative_edges = list(negative_edges)

# Create labels
edges = positive_edges + negative_edges
labels = [1] * len(positive_edges) + [0] * len(negative_edges)

# Convert to tensors
edges = torch.tensor(edges, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.float)

print("Total samples:", len(edges))
print("Positive:", len(positive_edges))
print("Negative:", len(negative_edges))