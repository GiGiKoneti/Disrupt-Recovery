"""
SynthDetect — Contrastive Encoder Training Notebook
====================================================
Run in Google Colab or Kaggle (needs GPU).

How to use:
  1. Upload this file to Colab / Kaggle
  2. Upload your feature CSVs (data/processed/features/)
  3. Run all cells

Or convert to .ipynb:
  jupytext --to notebook 02_train_contrastive_encoder.py

Author: SynthDetect Team
"""

# %% [markdown]
# # 🧠 SynthDetect — Contrastive Encoder Training
#
# Trains the FAID contrastive encoder using supervised contrastive loss.
#
# **Objective**: Learn an embedding space where texts from the same
# source (human, GPT-4, Claude, Gemini) cluster together.

# %% — Setup & Install Dependencies
# fmt: off
# !pip install torch scikit-learn pandas numpy matplotlib seaborn tqdm faiss-cpu

# fmt: on
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# %% — Configuration
CONFIG = {
    "input_dim": 76,          # Feature vector dimension (adjust based on your extracted features)
    "hidden_dims": [512, 256],
    "embedding_dim": 256,
    "dropout": 0.3,
    "temperature": 0.07,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 50,
    "batch_size": 64,
    "early_stopping_patience": 5,
    "train_split": 0.8,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
}

print(f"Device: {CONFIG['device']}")
print(f"Config: {json.dumps(CONFIG, indent=2)}")

# Set seed for reproducibility
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# %% — Model Definition
class ContrastiveEncoder(nn.Module):
    """MLP contrastive encoder for stylometric features."""

    def __init__(self, input_dim, hidden_dims, embedding_dim, dropout):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        embedding = self.encoder(x)
        return torch.nn.functional.normalize(embedding, p=2, dim=1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        similarity = torch.matmul(features, features.T) / self.temperature
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        mask_sum = torch.clamp(mask.sum(dim=1), min=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum
        return -mean_log_prob.mean()


# %% — Feature Dataset
class FeatureDataset(Dataset):
    """Dataset for pre-extracted feature vectors."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# %% — Generate Synthetic Training Data (for demo)
# In production, replace this with real extracted features from HC3/M4GT
def generate_demo_data(n_samples_per_class=500, feature_dim=76, seed=42):
    """
    Generate synthetic feature data for training demo.

    Creates features with different statistical properties per source
    to simulate what real feature extraction would produce.
    """
    np.random.seed(seed)
    sources = ["human", "gpt4", "claude", "gemini"]
    all_features = []
    all_labels = []

    # Each source gets slightly different feature distributions
    source_params = {
        "human": {"mean_shift": 0.0, "std": 1.2, "skew": 0.3},
        "gpt4": {"mean_shift": 0.5, "std": 0.8, "skew": -0.1},
        "claude": {"mean_shift": 0.3, "std": 0.9, "skew": 0.0},
        "gemini": {"mean_shift": 0.7, "std": 0.7, "skew": -0.2},
    }

    for source_idx, source in enumerate(sources):
        params = source_params[source]
        features = np.random.normal(
            loc=params["mean_shift"],
            scale=params["std"],
            size=(n_samples_per_class, feature_dim),
        )
        # Add source-specific patterns
        features[:, :10] += params["skew"] * np.random.randn(n_samples_per_class, 10)
        features[:, 10:20] *= (1 + 0.2 * source_idx)  # Systematic variation

        all_features.append(features)
        all_labels.extend([source_idx] * n_samples_per_class)

    return np.vstack(all_features).astype(np.float32), np.array(all_labels), sources


print("Generating demo training data...")
features, labels, source_names = generate_demo_data(
    n_samples_per_class=500,
    feature_dim=CONFIG["input_dim"],
)
print(f"  Total samples: {len(features)}")
print(f"  Feature dim: {features.shape[1]}")
print(f"  Sources: {source_names}")
print(f"  Label distribution: {dict(zip(source_names, np.bincount(labels)))}")


# %% — Create DataLoaders
dataset = FeatureDataset(features, labels)
train_size = int(CONFIG["train_split"] * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(CONFIG["seed"]),
)

train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

print(f"\n  Train: {len(train_dataset)} samples")
print(f"  Val:   {len(val_dataset)} samples")


# %% — Training Loop
model = ContrastiveEncoder(
    input_dim=CONFIG["input_dim"],
    hidden_dims=CONFIG["hidden_dims"],
    embedding_dim=CONFIG["embedding_dim"],
    dropout=CONFIG["dropout"],
).to(CONFIG["device"])

criterion = SupConLoss(temperature=CONFIG["temperature"])
optimizer = optim.Adam(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

# Early stopping
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None
train_losses = []
val_losses = []

print(f"\n{'='*60}")
print(f"  Training Contrastive Encoder")
print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"{'='*60}\n")

for epoch in range(1, CONFIG["epochs"] + 1):
    # === Train ===
    model.train()
    epoch_train_loss = 0
    n_batches = 0

    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(CONFIG["device"])
        batch_labels = batch_labels.to(CONFIG["device"])

        optimizer.zero_grad()
        embeddings = model(batch_features)
        loss = criterion(embeddings, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        n_batches += 1

    avg_train_loss = epoch_train_loss / n_batches
    train_losses.append(avg_train_loss)

    # === Validate ===
    model.eval()
    epoch_val_loss = 0
    n_val_batches = 0

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(CONFIG["device"])
            batch_labels = batch_labels.to(CONFIG["device"])

            embeddings = model(batch_features)
            loss = criterion(embeddings, batch_labels)

            epoch_val_loss += loss.item()
            n_val_batches += 1

    avg_val_loss = epoch_val_loss / n_val_batches
    val_losses.append(avg_val_loss)

    scheduler.step()

    # Log every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        lr = scheduler.get_last_lr()[0]
        print(
            f"  Epoch {epoch:3d}/{CONFIG['epochs']} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"LR: {lr:.6f}"
        )

    # Early stopping
    if avg_val_loss < best_val_loss - 0.001:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= CONFIG["early_stopping_patience"]:
            print(f"\n  ⏹ Early stopping at epoch {epoch} (patience={CONFIG['early_stopping_patience']})")
            break

print(f"\n  ✅ Training complete!")
print(f"  Best validation loss: {best_val_loss:.4f}")


# %% — Save Model Checkpoint
# Restore best model
if best_model_state:
    model.load_state_dict(best_model_state)

# Save checkpoint
save_dir = Path("data/models")
save_dir.mkdir(parents=True, exist_ok=True)
checkpoint_path = save_dir / "contrastive_encoder_v1.pth"

torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": best_val_loss,
    "config": CONFIG,
    "source_names": source_names,
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_loss": best_val_loss,
    },
}, str(checkpoint_path))

print(f"\n  💾 Checkpoint saved to: {checkpoint_path}")


# %% — Plot Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(train_losses, label="Train Loss", color="#2196F3", linewidth=2)
ax1.plot(val_losses, label="Val Loss", color="#FF5722", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Contrastive Loss")
ax1.set_title("Training & Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# t-SNE visualization of learned embeddings
model.eval()
all_embeddings = []
all_labels_list = []

with torch.no_grad():
    for batch_features, batch_labels in val_loader:
        batch_features = batch_features.to(CONFIG["device"])
        embeddings = model(batch_features)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels_list.append(batch_labels.numpy())

all_embeddings = np.vstack(all_embeddings)
all_labels_arr = np.concatenate(all_labels_list)

# t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Plot
colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]
for idx, source in enumerate(source_names):
    mask = all_labels_arr == idx
    ax2.scatter(
        embeddings_2d[mask, 0], embeddings_2d[mask, 1],
        c=colors[idx], label=source, alpha=0.6, s=20,
    )

ax2.set_title("t-SNE of Learned Embeddings")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig_path = Path("research/experiments/results")
fig_path.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path / "training_curves_tsne.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\n  📊 Plots saved to: {fig_path / 'training_curves_tsne.png'}")


# %% — Build FAISS Index from Trained Embeddings
import faiss

print("\n  Building FAISS index from training embeddings...")

# Encode all training data
model.eval()
train_embeddings = []
train_labels_list = []
train_metadata = []

with torch.no_grad():
    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(CONFIG["device"])
        embeddings = model(batch_features)
        train_embeddings.append(embeddings.cpu().numpy())
        train_labels_list.append(batch_labels.numpy())

train_embeddings = np.vstack(train_embeddings).astype(np.float32)
train_labels_arr = np.concatenate(train_labels_list)

# Create FAISS index
index = faiss.IndexFlatIP(CONFIG["embedding_dim"])
faiss.normalize_L2(train_embeddings)
index.add(train_embeddings)

# Save index
vector_db_dir = Path("data/vector_db")
vector_db_dir.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(vector_db_dir / "main_index.faiss"))

# Save metadata
metadata = [
    {"source": source_names[int(label)], "label": int(label)}
    for label in train_labels_arr
]
with open(vector_db_dir / "main_metadata.json", "w") as f:
    json.dump(metadata, f)

print(f"  ✅ FAISS index built: {index.ntotal} vectors")
print(f"  💾 Saved to: {vector_db_dir}")


# %% — Quick Validation: k-NN Attribution Test
print("\n  Running k-NN attribution validation...")

# Take a few validation samples and check attribution accuracy
correct = 0
total = 0

with torch.no_grad():
    for batch_features, batch_labels in val_loader:
        batch_features = batch_features.to(CONFIG["device"])
        embeddings = model(batch_features).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(embeddings)

        # Search
        k = 5
        distances, indices = index.search(embeddings, k)

        for i in range(len(embeddings)):
            # Get neighbor labels
            neighbor_labels = [int(train_labels_arr[idx]) for idx in indices[i]]
            # Majority vote
            from collections import Counter
            predicted = Counter(neighbor_labels).most_common(1)[0][0]
            actual = int(batch_labels[i])

            if predicted == actual:
                correct += 1
            total += 1

accuracy = correct / total if total > 0 else 0
print(f"  k-NN attribution accuracy (k=5): {accuracy:.1%}")
print(f"  ({correct}/{total} correct)")

# %% [markdown]
# ## Summary
#
# | Metric | Value |
# |--------|-------|
# | Best Val Loss | see above |
# | k-NN Accuracy | see above |
# | Index Size | see above |
#
# **Next Steps:**
# 1. Replace synthetic data with real HC3/M4GT extracted features
# 2. Fine-tune hyperparameters with Optuna
# 3. Download checkpoint and FAISS index to local machine
