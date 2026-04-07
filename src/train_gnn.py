import torch
import torch.optim as optim
from prepare_gnn_data import graph_data
from gnn_model import GNN

model = GNN(input_dim=16, hidden_dim=32, output_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    
    out = model(graph_data.x, graph_data.edge_index)
    loss = criterion(out, graph_data.y)
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")