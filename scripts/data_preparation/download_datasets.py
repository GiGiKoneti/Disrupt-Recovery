"""
Dataset Download Script

Downloads public datasets for training and evaluation:
- HC3 (Human ChatGPT Comparison Corpus) from Hugging Face
- M4GT-Bench (Multi-Generator, Multi-Domain) from Hugging Face

Author: SynthDetect Team
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def download_hc3(output_dir: str = "data/raw/hc3"):
    """Download HC3 dataset from Hugging Face."""
    try:
        from datasets import load_dataset

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("📥 Downloading HC3 dataset...")
        dataset = load_dataset("Hello-SimpleAI/HC3", trust_remote_code=True)

        # Save to disk
        dataset.save_to_disk(str(output_path))
        print(f"✅ HC3 saved to {output_path}")

        # Print stats
        for split in dataset:
            print(f"   {split}: {len(dataset[split])} samples")

        return dataset

    except Exception as e:
        print(f"❌ HC3 download failed: {e}")
        print("   Try: pip install datasets")
        return None


def download_m4gt(output_dir: str = "data/raw/m4gt"):
    """Download M4GT-Bench dataset from Hugging Face."""
    try:
        from datasets import load_dataset

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("📥 Downloading M4GT-Bench dataset...")
        # M4GT has multiple configs — download the main one
        dataset = load_dataset(
            "SemEval2024-MGTD/subtaskA",
            trust_remote_code=True,
        )

        dataset.save_to_disk(str(output_path))
        print(f"✅ M4GT-Bench saved to {output_path}")

        for split in dataset:
            print(f"   {split}: {len(dataset[split])} samples")

        return dataset

    except Exception as e:
        print(f"❌ M4GT download failed: {e}")
        print("   This dataset may require different access. Check HuggingFace.")
        return None


def main():
    """Download all datasets."""
    print("=" * 50)
    print("SynthDetect - Dataset Downloader")
    print("=" * 50)
    print()

    download_hc3()
    print()
    download_m4gt()

    print()
    print("=" * 50)
    print("Download complete! Check data/raw/ for datasets.")
    print("=" * 50)


if __name__ == "__main__":
    main()
