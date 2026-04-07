import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load encoded data
data = pd.read_csv("data/drug_adr_encoded.csv")

G = nx.Graph()

# Add edges between drug and ADR
for _, row in data.iterrows():
    G.add_edge(f"drug_{row['drug_id']}", f"adr_{row['adr_id']}")

print("Total nodes:", G.number_of_nodes())
print("Total edges:", G.number_of_edges())

# Draw small subgraph (for visualization)
plt.figure(figsize=(8,6))
sub_nodes = list(G.nodes())[:50]
subgraph = G.subgraph(sub_nodes)

nx.draw(subgraph, with_labels=False, node_size=50)
plt.show()