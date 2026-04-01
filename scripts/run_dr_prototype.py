#!/usr/bin/env python3
"""
SynthDetect — D&R Pipeline End-to-End Prototype
================================================

Validates the core research hypothesis:
  AI-generated text, when disrupted (shuffled) and recovered by an LLM,
  will be reconstructed with HIGHER fidelity than human-written text
  due to posterior concentration.

Pipeline: Input → Chunk → Shuffle → Gemini Recovery → Similarity → Score

Usage:
    poetry run python scripts/run_dr_prototype.py

Author: SynthDetect Team
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.utils.text_utils import clean_text, count_words
from src.utils.validators import validate_api_key
from src.dr_pipeline.chunking import ChunkingEngine, ChunkStrategy
from src.dr_pipeline.shuffling import ShuffleEngine, ShuffleLevel
from src.dr_pipeline.recovery import RecoveryEngine
from src.dr_pipeline.similarity import SimilarityEngine
from src.llm_integration.gemini_client import GeminiClient


# ============================================================
# Sample Texts for Hypothesis Testing
# ============================================================

HUMAN_TEXT = """
The old bookshop on Elm Street had survived three recessions, two floods, 
and the arrival of Amazon. Margaret Chen didn't know how, exactly. She 
suspected it had something to do with the cat. Whiskers — a enormous orange 
tabby of indeterminate age — had claimed the front window as his throne 
sometime around 2003, and customers kept coming back just to see him. "I 
only came in because of the cat," they'd say, then leave with four novels 
and a cookbook. Margaret had tried to retire twice. Both times, Whiskers had 
knocked her "CLOSING SALE" sign off the door. She took it as a sign — the 
metaphorical kind, not the cardboard one lying on the floor. Her daughter 
thought she was projecting. Maybe she was. But the shop was still open, 
the cat was still fat, and the books still smelled like promises. Some 
mornings, before the first customer arrived, she'd sit in the back room 
with her tea and listen to the building settle. It made sounds like an 
old ship, creaking and groaning with the weather. She'd built a life between 
these shelves, arranged and rearranged thousands of spines until the 
alphabet felt like a kind of music. A-B-C-D. Mystery, Romance, Science 
Fiction, Travel. Each section a different country she'd visited without 
ever leaving the shop.
"""

AI_TEXT = """
Artificial intelligence has fundamentally transformed the landscape of modern 
technology and its applications across diverse sectors. The rapid advancement 
of machine learning algorithms, particularly deep neural networks and 
transformer-based architectures, has enabled unprecedented capabilities in 
natural language processing, computer vision, and automated decision-making 
systems. These developments carry significant implications for industries 
ranging from healthcare and financial services to education and transportation. 
The integration of large language models into everyday applications has 
demonstrated remarkable proficiency in tasks such as text generation, 
summarization, and question answering. Furthermore, the emergence of 
multimodal AI systems capable of processing text, images, and audio 
simultaneously represents a paradigm shift in how we conceptualize 
human-computer interaction. As these technologies continue to evolve at 
an accelerating pace, it becomes increasingly important to establish 
comprehensive ethical frameworks and governance structures that ensure 
responsible development and deployment. The potential benefits of artificial 
intelligence are substantial and far-reaching, encompassing improved 
diagnostic accuracy in medical imaging, enhanced predictive modeling in 
climate science, and more efficient resource allocation in supply chain 
management. However, these advantages must be carefully balanced against 
legitimate concerns regarding algorithmic bias, data privacy, workforce 
displacement, and the concentration of technological power among a 
relatively small number of organizations.
"""


def print_banner():
    """Print a styled banner."""
    print("\n" + "=" * 70)
    print("  🔬 SynthDetect — D&R Pipeline End-to-End Prototype")
    print("  Validating the Posterior Concentration Hypothesis")
    print("=" * 70)
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  LLM       : {settings.DEFAULT_MODEL}")
    print(f"  Strategy  : Semantic Chunking → Sentence Shuffling → Gemini Recovery")
    print("=" * 70 + "\n")


def run_dr_on_text(
    text: str,
    label: str,
    chunking_engine: ChunkingEngine,
    shuffle_engine: ShuffleEngine,
    recovery_engine: RecoveryEngine,
    similarity_engine: SimilarityEngine,
    seed: int = 42,
):
    """
    Run the full D&R pipeline on a single text.

    Returns dict with per-chunk and aggregate results.
    """
    cleaned = clean_text(text)
    word_count = count_words(cleaned)

    print(f"\n{'─' * 60}")
    print(f"  📝 Processing: {label}")
    print(f"  Words: {word_count}")
    print(f"{'─' * 60}")

    # Step 1: Chunk
    chunks = chunking_engine.chunk_text(cleaned)
    print(f"\n  📦 Chunked into {len(chunks)} segments")

    chunk_results = []

    for i, chunk in enumerate(chunks):
        chunk_words = count_words(chunk)
        print(f"\n  ── Chunk {i + 1}/{len(chunks)} ({chunk_words} words) ──")

        # Step 2: Shuffle
        shuffled = shuffle_engine.disrupt(chunk, seed=seed + i)
        disruption = shuffle_engine.compute_disruption_score(chunk, shuffled)
        print(f"  🔀 Disruption score: {disruption:.3f}")

        # Show first 80 chars of original vs shuffled
        print(f"     Original : {chunk[:80]}...")
        print(f"     Shuffled : {shuffled[:80]}...")

        # Step 3: Recover via Gemini
        print(f"  🤖 Calling Gemini for recovery...", end=" ", flush=True)
        start_time = time.time()
        recovered = recovery_engine.recover(shuffled)
        latency = time.time() - start_time
        print(f"Done ({latency:.1f}s)")

        print(f"     Recovered: {recovered[:80]}...")

        # Step 4: Similarity (original vs recovered)
        similarity = similarity_engine.compute_similarity(
            original=chunk,
            recovered=recovered,
        )

        print(f"  📊 Similarity Scores:")
        print(f"     Semantic   : {similarity['semantic']:.4f}")
        print(f"     Structural : {similarity['structural']:.4f}")
        print(f"     Combined   : {similarity['combined']:.4f}")

        chunk_results.append({
            "chunk_index": i,
            "chunk_words": chunk_words,
            "disruption_score": disruption,
            "semantic_similarity": similarity["semantic"],
            "structural_similarity": similarity["structural"],
            "combined_similarity": similarity["combined"],
            "recovery_latency_s": round(latency, 2),
        })

    # Aggregate
    import numpy as np
    combined_scores = [cr["combined_similarity"] for cr in chunk_results]
    semantic_scores = [cr["semantic_similarity"] for cr in chunk_results]
    structural_scores = [cr["structural_similarity"] for cr in chunk_results]

    aggregate = {
        "label": label,
        "word_count": word_count,
        "num_chunks": len(chunks),
        "mean_combined": round(float(np.mean(combined_scores)), 4),
        "std_combined": round(float(np.std(combined_scores)), 4),
        "mean_semantic": round(float(np.mean(semantic_scores)), 4),
        "mean_structural": round(float(np.mean(structural_scores)), 4),
        "chunk_results": chunk_results,
    }

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  {label:^34s}  ║")
    print(f"  ╠══════════════════════════════════════╣")
    print(f"  ║  Mean Combined Sim : {aggregate['mean_combined']:.4f}            ║")
    print(f"  ║  Std Combined Sim  : {aggregate['std_combined']:.4f}            ║")
    print(f"  ║  Mean Semantic     : {aggregate['mean_semantic']:.4f}            ║")
    print(f"  ║  Mean Structural   : {aggregate['mean_structural']:.4f}            ║")
    print(f"  ╚══════════════════════════════════════╝")

    return aggregate


def print_comparison(human_result, ai_result):
    """Print side-by-side comparison and hypothesis validation."""
    print("\n" + "=" * 70)
    print("  📊 HYPOTHESIS VALIDATION — RESULTS COMPARISON")
    print("=" * 70)

    print(f"""
  ┌────────────────────┬──────────────┬──────────────┐
  │     Metric         │  👤 Human    │  🤖 AI       │
  ├────────────────────┼──────────────┼──────────────┤
  │ Word Count         │  {human_result['word_count']:>10}  │  {ai_result['word_count']:>10}  │
  │ Num Chunks         │  {human_result['num_chunks']:>10}  │  {ai_result['num_chunks']:>10}  │
  │ Mean Combined Sim  │  {human_result['mean_combined']:>10.4f}  │  {ai_result['mean_combined']:>10.4f}  │
  │ Std Combined Sim   │  {human_result['std_combined']:>10.4f}  │  {ai_result['std_combined']:>10.4f}  │
  │ Mean Semantic Sim  │  {human_result['mean_semantic']:>10.4f}  │  {ai_result['mean_semantic']:>10.4f}  │
  │ Mean Structural    │  {human_result['mean_structural']:>10.4f}  │  {ai_result['mean_structural']:>10.4f}  │
  └────────────────────┴──────────────┴──────────────┘
    """)

    # Hypothesis check
    delta = ai_result["mean_combined"] - human_result["mean_combined"]
    hypothesis_supported = delta > 0

    print(f"  Δ (AI - Human) Combined Similarity: {delta:+.4f}")
    print()

    if hypothesis_supported:
        print("  ✅ HYPOTHESIS SUPPORTED!")
        print("  AI-generated text recovered with HIGHER fidelity than human text.")
        print("  This is consistent with the posterior concentration theory:")
        print("  LLMs recover AI text better because it lies in high-probability regions")
        print("  of the model's output distribution.")
    else:
        print("  ❌ HYPOTHESIS NOT SUPPORTED (with these samples)")
        print("  Human text recovered with equal or higher fidelity.")
        print("  This may be due to:")
        print("  - Sample-specific effects (try more diverse texts)")
        print("  - Chunk size or shuffling parameters need tuning")
        print("  - The specific LLM used (Gemini) may behave differently")

    print(f"\n  Confidence delta: |Δ| = {abs(delta):.4f}")
    if abs(delta) < 0.02:
        print("  ⚠️  Delta is very small — results may not be statistically significant")
        print("     Need larger sample size for reliable conclusions")
    elif abs(delta) < 0.05:
        print("  📊 Delta is moderate — suggestive but needs confirmation with more data")
    else:
        print("  💪 Delta is substantial — strong signal for the hypothesis")

    return {
        "delta": round(delta, 4),
        "hypothesis_supported": hypothesis_supported,
        "human_score": human_result["mean_combined"],
        "ai_score": ai_result["mean_combined"],
    }


def save_results(human_result, ai_result, comparison, output_dir="data/experiments"):
    """Save results to JSON for future analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_path / f"dr_prototype_{timestamp}.json"

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": settings.DEFAULT_MODEL,
        "comparison": comparison,
        "human_result": human_result,
        "ai_result": ai_result,
    }

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  💾 Results saved to: {filepath}")
    return filepath


def main():
    """Run the end-to-end D&R prototype."""
    print_banner()

    # Validate API key
    is_valid, error = validate_api_key(settings.GOOGLE_API_KEY, "Google")
    if not is_valid:
        print(f"  ❌ {error}")
        print("  Set GOOGLE_API_KEY in your .env file.")
        sys.exit(1)

    print(f"  ✅ API key validated")

    # Initialize pipeline components
    print("\n  Initializing pipeline components...")

    gemini_client = GeminiClient()
    print(f"  ✅ Gemini client ready ({settings.DEFAULT_MODEL})")

    chunking_engine = ChunkingEngine(
        chunk_size=150,  # Smaller chunks for prototype (faster)
        strategy=ChunkStrategy.SEMANTIC,
    )
    print(f"  ✅ Chunking engine ready (semantic, 150 words/chunk)")

    shuffle_engine = ShuffleEngine(
        shuffle_level=ShuffleLevel.SENTENCE,
        preserve_ratio=0.2,
        preserve_boundaries=True,
    )
    print(f"  ✅ Shuffle engine ready (sentence-level, 20% preserved)")

    recovery_engine = RecoveryEngine(
        gemini_client=gemini_client,
        enable_cache=True,
    )
    print(f"  ✅ Recovery engine ready (caching enabled)")

    similarity_engine = SimilarityEngine(alpha=0.6)
    print(f"  ✅ Similarity engine ready (α=0.6)")

    # Run D&R on human text
    human_result = run_dr_on_text(
        text=HUMAN_TEXT,
        label="HUMAN-WRITTEN TEXT",
        chunking_engine=chunking_engine,
        shuffle_engine=shuffle_engine,
        recovery_engine=recovery_engine,
        similarity_engine=similarity_engine,
        seed=42,
    )

    # Run D&R on AI text
    ai_result = run_dr_on_text(
        text=AI_TEXT,
        label="AI-GENERATED TEXT",
        chunking_engine=chunking_engine,
        shuffle_engine=shuffle_engine,
        recovery_engine=recovery_engine,
        similarity_engine=similarity_engine,
        seed=42,
    )

    # Compare and validate hypothesis
    comparison = print_comparison(human_result, ai_result)

    # Save results
    save_results(human_result, ai_result, comparison)

    # Print LLM usage
    usage = gemini_client.usage_stats()
    print(f"\n  📈 LLM Usage:")
    print(f"     Total API calls  : {usage['total_calls']}")
    print(f"     Input tokens     : {usage['total_input_tokens']}")
    print(f"     Output tokens    : {usage['total_output_tokens']}")
    print(f"     Estimated cost   : ${usage['estimated_cost_usd']:.6f}")

    print("\n" + "=" * 70)
    print("  🏁 Prototype complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
