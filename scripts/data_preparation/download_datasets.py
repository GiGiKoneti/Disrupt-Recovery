#!/usr/bin/env python3
"""
Dataset Download Script
=======================

Downloads public datasets for training and evaluation:
- HC3 (Human ChatGPT Comparison Corpus) from Hugging Face
- M4GT-Bench (Multi-Generator, Multi-Domain) from Hugging Face
- TuringBench (Turing test benchmark) from Hugging Face

Also creates processed CSV files suitable for the FAID pipeline.

Usage:
    poetry run python scripts/data_preparation/download_datasets.py
    poetry run python scripts/data_preparation/download_datasets.py --dataset hc3
    poetry run python scripts/data_preparation/download_datasets.py --dataset m4gt
    poetry run python scripts/data_preparation/download_datasets.py --samples 500

Author: SynthDetect Team
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def download_hc3(output_dir: str = "data/raw/hc3", max_samples: int = 2000) -> pd.DataFrame | None:
    """
    Download HC3 dataset from Hugging Face.

    HC3 contains human answers and ChatGPT answers to the same questions
    across multiple domains (finance, medicine, open_qa, etc.).
    """
    try:
        from datasets import load_dataset

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("📥 Downloading HC3 dataset...")

        # HC3 has multiple configs — try the English "all" split
        configs_to_try = ["all", "finance", "medicine", "open_qa"]
        dataset = None

        for config in configs_to_try:
            try:
                print(f"   Trying config: {config}...")
                dataset = load_dataset(
                    "Hello-SimpleAI/HC3",
                    name=config,
                    trust_remote_code=True,
                )
                print(f"   ✅ Loaded config: {config}")
                break
            except Exception as e:
                print(f"   ⚠️  Config '{config}' failed: {str(e)[:80]}")
                continue

        if dataset is None:
            # Fallback: try without config
            try:
                dataset = load_dataset("Hello-SimpleAI/HC3", trust_remote_code=True)
            except Exception as e:
                print(f"❌ HC3 download failed: {e}")
                return None

        # Process into flat DataFrame
        rows = []
        for split_name in dataset:
            split = dataset[split_name]
            for item in split:
                question = item.get("question", "")

                # Human answers
                human_answers = item.get("human_answers", [])
                for ans in human_answers[:2]:  # Limit per question
                    if ans and len(ans.strip()) > 50:
                        rows.append({
                            "text": ans.strip(),
                            "label": "human",
                            "source": "human",
                            "domain": split_name,
                            "question": question[:200],
                        })

                # ChatGPT answers
                chatgpt_answers = item.get("chatgpt_answers", [])
                for ans in chatgpt_answers[:2]:
                    if ans and len(ans.strip()) > 50:
                        rows.append({
                            "text": ans.strip(),
                            "label": "ai",
                            "source": "chatgpt",
                            "domain": split_name,
                            "question": question[:200],
                        })

                if len(rows) >= max_samples:
                    break
            if len(rows) >= max_samples:
                break

        df = pd.DataFrame(rows)

        # Save
        csv_path = output_path / "hc3_processed.csv"
        df.to_csv(csv_path, index=False)

        print(f"✅ HC3 saved to {csv_path}")
        print(f"   Total samples: {len(df)}")
        print(f"   Human: {len(df[df['label'] == 'human'])}")
        print(f"   AI:    {len(df[df['label'] == 'ai'])}")

        return df

    except ImportError:
        print("❌ HC3 download failed: 'datasets' package not installed.")
        print("   Run: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ HC3 download failed: {e}")
        return None


def download_m4gt(output_dir: str = "data/raw/m4gt", max_samples: int = 2000) -> pd.DataFrame | None:
    """
    Download M4GT-Bench dataset from Hugging Face.

    This dataset is from SemEval-2024 Task 8 (Multigenerator Multidomain
    Machine-Generated Text Detection).
    """
    try:
        from datasets import load_dataset

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("📥 Downloading M4GT-Bench dataset...")

        # Try multiple known dataset IDs for this task
        dataset_ids = [
            "anyangsong/SemEval2024-Task8-SubtaskA",
            "SemEval2024-Task8/subtaskA",
            "SemEval2024-MGTD/subtaskA",
        ]

        dataset = None
        for ds_id in dataset_ids:
            try:
                print(f"   Trying: {ds_id}...")
                if ds_id == "anyangsong/SemEval2024-Task8-SubtaskA":
                    # This specific one needs a file-based load or correctly formatted config
                    dataset = load_dataset(ds_id, trust_remote_code=True)
                else:
                    dataset = load_dataset(ds_id, trust_remote_code=True)
                print(f"   ✅ Loaded: {ds_id}")
                break
            except Exception as e:
                print(f"   ⚠️  '{ds_id}' failed: {str(e)[:80]}")
                continue

        if dataset is None:
            print("❌ M4GT download failed: None of the dataset IDs were accessible.")
            return None

        # Process into flat DataFrame
        rows = []
        # Support both 'train' split and other potential splits
        splits = list(dataset.keys())
        for split_name in splits:
            split = dataset[split_name]
            for item in split:
                text = item.get("text", "") or item.get("content", "")
                label_raw = item.get("label", None)

                if not text or len(text.strip()) < 50:
                    continue

                # Normalize labels
                if isinstance(label_raw, int):
                    label = "ai" if label_raw == 1 else "human"
                elif isinstance(label_raw, str):
                    label = "ai" if label_raw.lower() in ["machine", "ai", "generated", "1"] else "human"
                else:
                    continue

                source = item.get("model", item.get("source", "unknown"))

                rows.append({
                    "text": text.strip()[:2000],  # Cap length
                    "label": label,
                    "source": str(source).lower(),
                    "domain": item.get("domain", split_name),
                })

                if len(rows) >= max_samples:
                    break
            if len(rows) >= max_samples:
                break

        df = pd.DataFrame(rows)

        csv_path = output_path / "m4gt_processed.csv"
        df.to_csv(csv_path, index=False)

        print(f"✅ M4GT saved to {csv_path}")
        print(f"   Total samples: {len(df)}")
        if len(df) > 0:
            print(f"   Human: {len(df[df['label'] == 'human'])}")
            print(f"   AI:    {len(df[df['label'] == 'ai'])}")
            if "source" in df.columns:
                print(f"   Sources: {df['source'].value_counts().to_dict()}")

        return df

    except ImportError:
        print("❌ M4GT download failed: 'datasets' package not installed.")
        return None
    except Exception as e:
        print(f"❌ M4GT download failed: {e}")
        return None


def download_turingbench(output_dir: str = "data/raw/turingbench", max_samples: int = 2000) -> pd.DataFrame | None:
    """
    Download TuringBench dataset — a multi-generator benchmark with
    texts from 19 different generator models.
    """
    try:
        from datasets import load_dataset

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("📥 Downloading OpenTuringBench dataset...")
        # Using OpenTuringBench in_domain config as it is stable and comprehensive
        dataset = load_dataset("MLNTeam-Unical/OpenTuringBench", "in_domain", trust_remote_code=True)

        rows = []
        for split_name in dataset:
            split = dataset[split_name]
            for item in split:
                text = item.get("Generation", "") or item.get("text", "")
                model = item.get("model", "") or item.get("source", "")

                if not text or len(text.strip()) < 50:
                    continue

                label = "human" if "human" in str(model).lower() else "ai"

                rows.append({
                    "text": text.strip()[:2000],
                    "label": label,
                    "source": str(model).lower(),
                    "domain": "news",
                })

                if len(rows) >= max_samples:
                    break
            if len(rows) >= max_samples:
                break

        df = pd.DataFrame(rows)
        csv_path = output_path / "turingbench_processed.csv"
        df.to_csv(csv_path, index=False)

        print(f"✅ TuringBench saved to {csv_path}")
        print(f"   Total: {len(df)}, Human: {len(df[df['label']=='human'])}, AI: {len(df[df['label']=='ai'])}")

        return df

    except Exception as e:
        print(f"⚠️  TuringBench download failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="SynthDetect Dataset Downloader")
    parser.add_argument("--dataset", choices=["hc3", "m4gt", "turingbench", "all"], default="all")
    parser.add_argument("--samples", type=int, default=2000, help="Max samples per dataset")
    args = parser.parse_args()

    print("=" * 60)
    print("  SynthDetect — Dataset Downloader")
    print("=" * 60)
    print()

    results = {}

    if args.dataset in ("hc3", "all"):
        results["hc3"] = download_hc3(max_samples=args.samples)
        print()

    if args.dataset in ("m4gt", "all"):
        results["m4gt"] = download_m4gt(max_samples=args.samples)
        print()

    if args.dataset in ("turingbench", "all"):
        results["turingbench"] = download_turingbench(max_samples=args.samples)
        print()

    # Summary
    print("=" * 60)
    print("  📊 Download Summary")
    print("=" * 60)

    for name, df in results.items():
        if df is not None:
            print(f"  ✅ {name:15s}: {len(df)} samples")
        else:
            print(f"  ❌ {name:15s}: failed")

    print("=" * 60)


if __name__ == "__main__":
    main()
