import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ieee_cis" / "processed" / "train_features.csv"
MODEL_DIR = ROOT / "models" / "ieee_cis"
FEAST_FILE = ROOT / "feature_repo" / "feature_repo" / "data" / "fraud_features.parquet"

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv(DATA_PATH)
y = df["isFraud"].astype(int).to_numpy()
train_idx, _ = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
)
train_df = df.iloc[train_idx].copy()
print("Training transactions:", len(train_df))

card_cols = ["card1", "card2", "card3", "card5", "card6"]
keys = train_df[card_cols].fillna("unknown").astype(str).agg("_".join, axis=1)
missing = train_df[card_cols].isna().all(axis=1)
keys.loc[missing] = "unknown_card_" + train_df.loc[missing, "TransactionID"].astype(str)
train_df["card_node"] = "card_" + keys
train_df["device_node"] = "device_" + train_df["DeviceInfo"].fillna("unknown").astype(str)

all_nodes = pd.Index(pd.concat([train_df["card_node"], train_df["device_node"]]).unique())
node_to_id = {n: i for i, n in enumerate(all_nodes)}
card_ids = train_df["card_node"].map(node_to_id).to_numpy()
device_ids = train_df["device_node"].map(node_to_id).to_numpy()
edge_index = torch.tensor(
    np.vstack([np.concatenate([card_ids, device_ids]),
               np.concatenate([device_ids, card_ids])]),
    dtype=torch.long,
)

feats = ["TransactionAmt", "card_txn_count_1h", "card_txn_count_24h",
         "card_avg_amount_30d", "device_changed", "address_changed", "device_missing"]
stats = train_df.groupby("card_node")[feats].mean()
stats["transaction_count"] = train_df.groupby("card_node").size()
stats = stats.replace([np.inf, -np.inf], np.nan)
stats = stats.fillna(stats.median()).fillna(0)
stats = (stats - stats.mean()) / stats.std().replace(0, 1)

x = torch.zeros((len(all_nodes), 9), dtype=torch.float32)
x[:, 0] = torch.tensor([1.0 if str(n).startswith("card_") else 0.0 for n in all_nodes])
for card, row in stats.iterrows():
    x[node_to_id[card], 1:] = torch.tensor(row.to_numpy(dtype=np.float32))

labels = torch.full((len(all_nodes),), -1, dtype=torch.long)
for card, lab in train_df.groupby("card_node")["isFraud"].max().items():
    labels[node_to_id[card]] = int(lab)

graph = Data(x=x, edge_index=edge_index, y=labels)

class FraudGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(9, 16)
        self.conv2 = GCNConv(16, 16)
        self.output = torch.nn.Linear(16, 2)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.output(x)

model = FraudGCN()
mask = graph.y >= 0
counts = torch.bincount(graph.y[mask], minlength=2).float()
weights = counts.sum() / (2 * counts)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("Training training-only GNN...")
for epoch in range(1, 101):
    model.train()
    opt.zero_grad()
    loss = F.cross_entropy(model(graph.x, graph.edge_index)[mask], graph.y[mask], weight=weights)
    loss.backward()
    opt.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    node_scores = torch.softmax(model(graph.x, graph.edge_index), dim=1)[:, 1].numpy()

out = {}
for node, idx in node_to_id.items():
    if node.startswith("card_") and not node.startswith("card_unknown_card_"):
        key = node[len("card_"):]
        s = round(float(node_scores[idx]), 6)
        out[key] = s
        if "unknown" in key:
            out[key.replace("unknown", "missing")] = s

path = MODEL_DIR / "gnn_card_scores.json"
path.write_text(json.dumps(out), encoding="utf-8")
print("Saved", len(out), "card scores to", path)
print("Sample GNN keys:", list(out)[:3])

try:
    feast = pd.read_parquet(FEAST_FILE)
    fk = set(feast["card_key"].astype(str))
    print("Sample Feast keys:", list(fk)[:3])
    print("GNN keys that also exist in Feast:", len(fk & set(out)), "of", len(out))
except Exception as e:
    print("Feast key check failed:", e)
