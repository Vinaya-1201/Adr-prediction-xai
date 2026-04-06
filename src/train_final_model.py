from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from final_multimodal_model import FinalADRModel
from prepare_gnn_data import graph_data

# -----------------------------
# 1. Load Datasets
# -----------------------------
ehr = pd.read_csv("data/data-ori.csv")
combined = pd.read_csv("data/combined_drug_adr.csv")

# -----------------------------
# 2. Encode Drug Names
# -----------------------------
drug_encoder = LabelEncoder()
drug_encoder.fit(combined["drug_name"])

drug_list = combined["drug_name"].unique()
ehr["drug_name"] = np.random.choice(drug_list, size=len(ehr))
ehr["drug_id"] = drug_encoder.transform(ehr["drug_name"])

# Encode Sex
ehr["SEX"] = ehr["SEX"].map({"M": 0, "F": 1})

# -----------------------------
# 3. Simulate ADR Outcome
# -----------------------------
ehr["adr_occurred"] = (
    (ehr["AGE"] > 65) |
    (ehr["LEUCOCYTE"] > 15) |
    (ehr["THROMBOCYTE"] > 400) |
    (ehr["HAEMOGLOBINS"] < 10)
).astype(int)

# -----------------------------
# 4. Prepare Tensors
# -----------------------------
lab_features = torch.tensor(
    ehr[[
        "HAEMATOCRIT",
        "HAEMOGLOBINS",
        "ERYTHROCYTE",
        "LEUCOCYTE",
        "THROMBOCYTE",
        "MCH",
        "MCHC",
        "MCV",
        "AGE",
        "SEX"
    ]].values,
    dtype=torch.float
)

labels = torch.tensor(ehr["adr_occurred"].values, dtype=torch.float)
drug_ids = torch.tensor(ehr["drug_id"].values, dtype=torch.long)

# -----------------------------
# 5. Train/Test Split (80/20)
# -----------------------------
train_idx, test_idx = train_test_split(
    range(len(labels)),
    test_size=0.2,
    random_state=42
)

train_idx = torch.tensor(train_idx)
test_idx = torch.tensor(test_idx)

# -----------------------------
# 6. Initialize Model
# -----------------------------
model = FinalADRModel(
    num_nodes=graph_data.x.shape[0],
    lab_dim=10
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

# -----------------------------
# 7. Training Loop
# -----------------------------
for epoch in range(20):

    model.train()
    optimizer.zero_grad()

    preds = model(
        graph_data.edge_index,
        drug_ids[train_idx],
        lab_features[train_idx]
    )

    loss = criterion(preds, labels[train_idx])
    loss.backward()
    optimizer.step()

    # Training Accuracy
    preds_binary = (preds > 0.5).float()
    train_acc = (
        (preds_binary == labels[train_idx]).sum().item()
        / len(train_idx)
    )

    print(
        f"Epoch {epoch+1}, "
        f"Loss: {loss.item():.4f}, "
        f"Train Acc: {train_acc*100:.2f}%"
    )
# -----------------------------
# Testing
# -----------------------------
model.eval()
with torch.no_grad():

    test_preds = model(
        graph_data.edge_index,
        drug_ids[test_idx],
        lab_features[test_idx]
    )

    test_probs = test_preds.numpy()
    true_labels = labels[test_idx].numpy()

# ---- ROC Curve ----
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(true_labels, test_probs)

# Find best threshold (Youden’s J statistic)
best_index = (tpr - fpr).argmax()
best_threshold = thresholds[best_index]

print(f"Best Threshold: {best_threshold:.4f}")

# Apply best threshold
test_binary = (test_probs > best_threshold).astype(int)

# ---- Metrics ----
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

accuracy = accuracy_score(true_labels, test_binary)
precision = precision_score(true_labels, test_binary)
recall = recall_score(true_labels, test_binary)
f1 = f1_score(true_labels, test_binary)
roc_auc = roc_auc_score(true_labels, test_probs)
cm = confusion_matrix(true_labels, test_binary)

print("\n===== Evaluation Metrics =====")
print(f"Accuracy:  {accuracy*100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# 9. Save Model
# -----------------------------
torch.save(model.state_dict(), "final_multimodal_model.pth")
print("Model saved successfully.")
# ----------------------------
# SAVE GRAPH FOR BACKEND
# ----------------------------

import torch

torch.save(graph_data.edge_index, "edge_index.pt")
torch.save(graph_data.x.shape[0], "num_nodes.pt")

print("Graph files saved successfully.")
import joblib

joblib.dump(drug_encoder, "backend/model/drug_encoder.pkl")