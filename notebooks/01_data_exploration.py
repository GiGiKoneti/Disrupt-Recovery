"""
SynthDetect — Data Exploration Notebook
=========================================
Run locally or convert to .ipynb:
  jupytext --to notebook 01_data_exploration.py

Author: SynthDetect Team
"""

# %% [markdown]
# # 📊 SynthDetect — Data Exploration
#
# This notebook provides an overview of the downloaded datasets used
# for training and evaluating the SynthDetect detection system.

# %% — Imports
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.text_utils import clean_text, count_words
from src.faid_pipeline.feature_extraction import FeatureExtractor

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.family"] = "sans-serif"

# %% — Load Datasets
data_dir = Path("data/raw")
csvs = sorted(data_dir.rglob("*_processed.csv"))

if not csvs:
    print("❌ No datasets found! Run download_datasets.py first:")
    print("   poetry run python scripts/data_preparation/download_datasets.py")
    sys.exit(0)

print(f"Found {len(csvs)} dataset files:")
for csv in csvs:
    print(f"  {csv}")

# Load and concatenate all datasets
dfs = []
for csv in csvs:
    df = pd.read_csv(csv)
    df["dataset"] = csv.stem.replace("_processed", "")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"\nTotal samples: {len(data)}")
print(f"Columns: {list(data.columns)}")

# %% — Basic Statistics
print("\n" + "=" * 60)
print("  📊 Dataset Overview")
print("=" * 60)

# Label distribution
print("\nLabel Distribution:")
print(data["label"].value_counts().to_string())

# Dataset breakdown
print("\nSamples per Dataset:")
print(data.groupby(["dataset", "label"]).size().unstack(fill_value=0).to_string())

# Source distribution (top 10)
if "source" in data.columns:
    print("\nTop 10 Sources:")
    print(data["source"].value_counts().head(10).to_string())

# %% — Text Length Analysis
data["word_count"] = data["text"].apply(lambda x: count_words(str(x)))
data["char_count"] = data["text"].apply(lambda x: len(str(x)))

print("\nText Length Statistics:")
print(data.groupby("label")["word_count"].describe().round(1).to_string())

# %% — Visualization: Length Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Word count distribution
for label in ["human", "ai"]:
    subset = data[data["label"] == label]["word_count"]
    axes[0].hist(subset, bins=50, alpha=0.6, label=f"{label} (n={len(subset)})")

axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Word Count Distribution by Label")
axes[0].legend()
axes[0].axvline(x=200, color="red", linestyle="--", alpha=0.5, label="Min for D&R")

# Box plot by source
if "source" in data.columns:
    top_sources = data["source"].value_counts().head(6).index.tolist()
    plot_data = data[data["source"].isin(top_sources)]
    sns.boxplot(data=plot_data, x="source", y="word_count", ax=axes[1])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_title("Word Count by Source")
else:
    sns.boxplot(data=data, x="label", y="word_count", ax=axes[1])
    axes[1].set_title("Word Count by Label")

plt.tight_layout()
fig_path = Path("research/experiments/results")
fig_path.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path / "data_exploration_lengths.png", dpi=150, bbox_inches="tight")
print(f"\n📊 Saved: {fig_path / 'data_exploration_lengths.png'}")

# %% — Feature Preview
print("\n" + "=" * 60)
print("  🔬 Feature Extraction Preview (5 samples per label)")
print("=" * 60)

extractor = FeatureExtractor()

for label in ["human", "ai"]:
    subset = data[data["label"] == label].head(5)
    print(f"\n--- {label.upper()} ---")
    for _, row in subset.iterrows():
        text = str(row["text"])[:500]
        features = extractor.extract_features(text)
        flat = extractor.extract_flat_vector(str(row["text"])[:500])
        print(f"  [{row.get('source', '?'):>10}] "
              f"words={count_words(text):>4} | "
              f"features={len(flat)} dims | "
              f"TTR={features['lexical'][0]:.3f} | "
              f"avg_word={features['lexical'][1]:.1f}")

# %% — Feature Distribution Comparison
print("\n" + "=" * 60)
print("  📈 Feature Distribution Comparison")
print("=" * 60)

# Extract features for a subset
n_preview = min(50, len(data) // 2)
human_subset = data[data["label"] == "human"].sample(n=n_preview, random_state=42)
ai_subset = data[data["label"] == "ai"].sample(n=min(n_preview, len(data[data["label"] == "ai"])), random_state=42)

human_features = []
ai_features = []

for _, row in human_subset.iterrows():
    feat = extractor.extract_features(str(row["text"])[:500])
    human_features.append({
        "type_token_ratio": feat["lexical"][0],
        "avg_word_length": feat["lexical"][1],
        "avg_sentence_length": feat["stylometric"][0]
    })

for _, row in ai_subset.iterrows():
    feat = extractor.extract_features(str(row["text"])[:500])
    ai_features.append({
        "type_token_ratio": feat["lexical"][0],
        "avg_word_length": feat["lexical"][1],
        "avg_sentence_length": feat["stylometric"][0]
    })

# Compare key lexical features
human_df = pd.DataFrame(human_features)
ai_df = pd.DataFrame(ai_features)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, col, title in zip(axes, ["type_token_ratio", "avg_word_length", "avg_sentence_length"],
                                ["Type-Token Ratio", "Avg Word Length", "Avg Sentence Length"]):
    if col in human_df.columns and col in ai_df.columns:
        ax.hist(human_df[col].dropna(), bins=20, alpha=0.6, label="Human", color="#4CAF50")
        ax.hist(ai_df[col].dropna(), bins=20, alpha=0.6, label="AI", color="#FF5722")
        ax.set_title(title)
        ax.legend()

plt.tight_layout()
plt.savefig(fig_path / "data_exploration_features.png", dpi=150, bbox_inches="tight")
print(f"\n📊 Saved: {fig_path / 'data_exploration_features.png'}")

# %% — Summary
print("\n" + "=" * 60)
print("  ✅ Data Exploration Complete")
print("=" * 60)
print(f"  Total samples: {len(data)}")
print(f"  Human: {len(data[data['label'] == 'human'])}")
print(f"  AI: {len(data[data['label'] == 'ai'])}")
print(f"  Mean word count: {data['word_count'].mean():.0f}")
print(f"  Texts >= 200 words: {len(data[data['word_count'] >= 200])}")
print(f"  Figures saved to: {fig_path}")
print("=" * 60)
