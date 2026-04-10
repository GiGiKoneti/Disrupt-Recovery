#!/usr/bin/env python3
"""
Build FAISS Vector Database
============================

Extracts features from downloaded datasets, encodes them with the
trained contrastive encoder, and builds a FAISS index for k-NN
attribution in the FAID pipeline.

Prerequisites:
  1. Download datasets: poetry run python scripts/data_preparation/download_datasets.py
  2. Train encoder:     Run notebooks/02_train_contrastive_encoder.py on Colab

Usage:
    poetry run python scripts/data_preparation/build_vector_db.py
    poetry run python scripts/data_preparation/build_vector_db.py --checkpoint data/models/contrastive_encoder_v1.pth
    poetry run python scripts/data_preparation/build_vector_db.py --max-samples 500

Author: SynthDetect Team
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def find_dataset_csvs(data_dir: str = "data/raw") -> list[Path]:
    """Find all processed CSV files from dataset downloads."""
    data_path = Path(data_dir)
    csvs = sorted(data_path.rglob("*_processed.csv"))
    return csvs


def extract_features_from_csv(
    csv_path: Path,
    max_samples: int = 1000,
) -> tuple[np.ndarray, list[dict]]:
    """
    Extract FAID features from a dataset CSV.

    Returns:
        features: (N, D) array of feature vectors
        metadata: List of dicts with text metadata
    """
    from src.faid_pipeline.feature_extraction import FeatureExtractor

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from {csv_path.name}")

    # Subsample if needed
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {max_samples} rows")

    extractor = FeatureExtractor()
    features_list = []
    metadata_list = []
    skipped = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  Extracting features"):
        text = str(row.get("text", ""))
        if len(text.strip()) < 30:
            skipped += 1
            continue

        try:
            feat_dict = extractor.extract_features(text)
            flat = extractor.extract_flat_vector(text)
            features_list.append(flat)

            metadata_list.append({
                "label": row.get("label", "unknown"),
                "source": row.get("source", "unknown"),
                "domain": row.get("domain", "unknown"),
                "text_preview": text[:100],
                "dataset": csv_path.stem,
            })
        except Exception as e:
            skipped += 1
            continue

    if skipped > 0:
        print(f"  Skipped {skipped} samples (too short or extraction error)")

    if not features_list:
        return np.array([]), []

    return np.array(features_list, dtype=np.float32), metadata_list


def build_faiss_index(
    features: np.ndarray,
    metadata: list[dict],
    output_dir: str = "data/vector_db",
    checkpoint_path: str | None = None,
):
    """
    Build FAISS index from features. Uses the contrastive encoder
    checkpoint if available, otherwise builds a raw feature index.
    """
    import faiss

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if checkpoint_path and Path(checkpoint_path).exists():
        # Encode through contrastive model
        import torch
        from src.models.encoder_network import ContrastiveEncoderNetwork

        print(f"\n  Loading encoder checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})

        model = ContrastiveEncoderNetwork(
            input_dim=features.shape[1],
            hidden_dims=config.get("hidden_dims", [512, 256]),
            embedding_dim=config.get("embedding_dim", 256),
            dropout=0.0,  # No dropout at inference
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print("  Encoding features through contrastive model...")
        with torch.no_grad():
            tensor = torch.FloatTensor(features)
            embeddings = model(tensor).numpy()

        dim = embeddings.shape[1]
        print(f"  Embedding dim: {dim}")
    else:
        print("\n  ⚠️  No encoder checkpoint found — building raw feature index")
        print("     Train the encoder first for better attribution accuracy")
        embeddings = features
        dim = features.shape[1]

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create index
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim after normalization
    index.add(embeddings)

    # Save index
    index_path = output_path / "main_index.faiss"
    faiss.write_index(index, str(index_path))

    # Save metadata
    meta_path = output_path / "main_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  ✅ FAISS index built successfully!")
    print(f"     Vectors: {index.ntotal}")
    print(f"     Dimension: {dim}")
    print(f"     Index saved to: {index_path}")
    print(f"     Metadata saved to: {meta_path}")

    # Print label distribution
    label_counts = {}
    source_counts = {}
    for m in metadata:
        label_counts[m["label"]] = label_counts.get(m["label"], 0) + 1
        source_counts[m["source"]] = source_counts.get(m["source"], 0) + 1

    print(f"\n  Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"     {label}: {count}")

    print(f"\n  Source distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"     {source}: {count}")

    return index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS Vector DB for FAID")
    parser.add_argument(
        "--checkpoint", type=str, default="data/models/contrastive_encoder_v1.pth",
        help="Path to trained contrastive encoder checkpoint",
    )
    parser.add_argument(
        "--max-samples", type=int, default=1000,
        help="Max samples per dataset CSV",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Directory containing dataset CSVs",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/vector_db",
        help="Output directory for FAISS index",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SynthDetect — Build FAISS Vector Database")
    print("=" * 60)

    # Find dataset CSVs
    csvs = find_dataset_csvs(args.data_dir)
    if not csvs:
        print(f"\n  ❌ No processed CSVs found in {args.data_dir}/")
        print("     Run download_datasets.py first:")
        print("     poetry run python scripts/data_preparation/download_datasets.py")
        sys.exit(1)

    print(f"\n  Found {len(csvs)} dataset files:")
    for csv in csvs:
        print(f"     {csv}")

    # Extract features from all CSVs
    all_features = []
    all_metadata = []

    for csv_path in csvs:
        print(f"\n  Processing: {csv_path.name}")
        features, metadata = extract_features_from_csv(csv_path, max_samples=args.max_samples)

        if len(features) > 0:
            all_features.append(features)
            all_metadata.extend(metadata)
            print(f"  → {len(features)} feature vectors extracted")
        else:
            print(f"  → No features extracted (skipped)")

    if not all_features:
        print("\n  ❌ No features extracted from any dataset!")
        sys.exit(1)

    # Combine
    combined_features = np.vstack(all_features)
    print(f"\n  Combined: {combined_features.shape[0]} vectors × {combined_features.shape[1]} dims")

    # Save extracted features as backup
    features_dir = Path("data/processed/features")
    features_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_dir / "all_features.npy", combined_features)

    meta_df = pd.DataFrame(all_metadata)
    meta_df.to_csv(features_dir / "all_metadata.csv", index=False)
    print(f"  Features saved to: {features_dir}")

    # Build FAISS index
    checkpoint = args.checkpoint if Path(args.checkpoint).exists() else None
    build_faiss_index(
        combined_features, all_metadata,
        output_dir=args.output_dir,
        checkpoint_path=checkpoint,
    )

    print("\n" + "=" * 60)
    print("  🏁 Vector DB build complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
