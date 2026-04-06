import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from prepare_gnn_data import graph_data
from link_prediction_data import edges, labels
from link_prediction_model import LinkPredictor

# ---------------------------
# Train/Test Split
# ---------------------------
edges_train, edges_test, labels_train, labels_test = train_test_split(
    edges, labels, test_size=0.2, random_state=42
)

# ---------------------------
# Build TRAIN graph only (NO DATA LEAKAGE)
# ---------------------------
train_positive_edges = edges_train[labels_train == 1]

edge_list = []
for d, a in train_positive_edges:
    edge_list.append([d.item(), a.item()])
    edge_list.append([a.item(), d.item()])

train_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# ---------------------------
# Initialize Model
# ---------------------------
num_nodes = graph_data.x.shape[0]
model = LinkPredictor(num_nodes=num_nodes, embed_dim=32, hidden_dim=64)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCELoss()

# ---------------------------
# Training Loop
# ---------------------------
for epoch in range(10):
    model.train()
    optimizer.zero_grad()

    preds = model(train_edge_index, edges_train)
    loss = criterion(preds, labels_train)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---------------------------
# Evaluation
# ---------------------------
model.eval()
with torch.no_grad():
    preds_test = model(train_edge_index, edges_test)
    preds_binary = (preds_test > 0.5).float()
    accuracy = (preds_binary == labels_test).sum().item() / len(labels_test)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")