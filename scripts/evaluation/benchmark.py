#!/usr/bin/env python3
"""
SynthDetect — Benchmark Evaluation Script
==========================================

Runs the D&R pipeline on a batch of texts and computes
detection performance metrics (AUROC, F1, Precision, Recall).

Supports two modes:
  1. Built-in samples: Quick validation with hardcoded examples
  2. Dataset mode: Evaluate on HC3/M4GT data (requires download first)

Usage:
    poetry run python scripts/evaluation/benchmark.py
    poetry run python scripts/evaluation/benchmark.py --samples 20
    poetry run python scripts/evaluation/benchmark.py --dataset hc3

Author: SynthDetect Team
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import settings
from src.dr_pipeline.chunking import ChunkingEngine, ChunkStrategy
from src.dr_pipeline.shuffling import ShuffleEngine
from src.dr_pipeline.recovery import RecoveryEngine
from src.dr_pipeline.similarity import SimilarityEngine
from src.llm_integration.gemini_client import GeminiClient
from src.utils.metrics import compute_binary_metrics
from src.utils.text_utils import clean_text, count_words


# ============================================================
# Built-in evaluation samples
# ============================================================

HUMAN_SAMPLES = [
    """My grandmother's kitchen was a disaster zone of flour and good intentions. 
    Every Sunday she'd attempt something from that crumbling Betty Crocker cookbook 
    — the one with the coffee stain on page 47 that she swore was there when she 
    bought it. The cookies always came out wrong. Too flat, too thick, burnt on the 
    bottom. But we ate them anyway, sitting on those wobbly stools she refused to 
    replace. "They've got character," she'd say, which was her way of saying she 
    was too stubborn to spend money on furniture. I miss those burnt cookies more 
    than I can explain. The taste of imperfection, I suppose. Something no 
    algorithm could ever replicate.""",

    """Running through the park last Tuesday, I stepped in a puddle so deep 
    it swallowed my shoe. Just consumed it. I stood there, one-shoed, rain 
    hammering down, watching a duck that seemed to be laughing at me. An old 
    man on a bench gave me a thumbs up, which felt deeply sarcastic. I limped 
    home, leaving wet footprints on the sidewalk like some sort of pathetic 
    Cinderella. My roommate asked what happened and I said "I was attacked" 
    which technically wasn't false if you count puddles as aggressive. The shoe 
    is still there, probably. A monument to my judgment.""",

    """The problem with teaching high school biology is that teenagers think 
    they already know everything about the human body because they have one. 
    "I don't need to study this, I literally am this," said Marcus from the 
    back row, and honestly he had a point. But then he couldn't name a single 
    organ besides the heart and brain, which tells you everything. I've been 
    teaching for twenty-two years. The curriculum changes, the textbooks get 
    shinier, but the look of absolute betrayal when students discover they 
    have to memorize the Krebs cycle — that never changes. It's the one 
    constant in biology education.""",

    """I found a letter in my father's desk drawer after he died. It was 
    addressed to someone named Claire, which wasn't my mother's name. For 
    three weeks I carried it in my jacket pocket, unopened, feeling its weight 
    like a small stone against my ribs. I told myself I was waiting for the 
    right moment, but really I was afraid. Of what? That he was someone I 
    didn't know? He was already that — death makes strangers of everyone. 
    I eventually opened it. It was a letter to his sister who died young. 
    Apologizing for not visiting more. I put it back in the desk.""",

    """There's a coffee shop near my apartment that plays exclusively 90s 
    R&B. I don't know whose decision that was but I respect it deeply. I go 
    there to write, mostly, though what I actually do is stare at my laptop 
    while Boyz II Men plays in the background and wonder if this is what 
    being a writer is. The barista knows my order, which makes me feel 
    simultaneously special and predictable. A double oat latte, no foam. 
    The same thing, every day. There's comfort in repetition, I tell myself. 
    Or maybe I'm just boring. Either way, the coffee is good.""",
]

AI_SAMPLES = [
    """Artificial intelligence has emerged as one of the most transformative 
    technologies of the twenty-first century, fundamentally reshaping industries, 
    economies, and social structures across the globe. The development of 
    sophisticated machine learning algorithms, particularly deep neural networks 
    and large language models, has enabled unprecedented capabilities in natural 
    language understanding, computer vision, and predictive analytics. These 
    advancements have significant implications for healthcare delivery, where 
    AI-powered diagnostic tools are achieving accuracy rates comparable to or 
    exceeding those of experienced clinicians in specific domains such as 
    radiology and pathology.""",

    """The concept of sustainability has become increasingly central to global 
    policy discussions and corporate strategy in recent decades. Environmental 
    sustainability encompasses the responsible management of natural resources, 
    the reduction of carbon emissions, and the preservation of biodiversity for 
    future generations. Organizations worldwide are implementing comprehensive 
    environmental, social, and governance frameworks to address stakeholder 
    concerns and regulatory requirements. The transition to renewable energy 
    sources, including solar, wind, and hydroelectric power, represents a 
    critical component of the broader sustainability agenda and requires 
    substantial investment in infrastructure modernization.""",

    """The evolution of digital communication technologies has fundamentally 
    altered the dynamics of human interaction and information exchange. Social 
    media platforms have created unprecedented opportunities for connection, 
    collaboration, and content sharing across geographical boundaries. However, 
    these platforms also present significant challenges related to misinformation 
    propagation, privacy concerns, and the potential for algorithmic amplification 
    of divisive content. Research indicates that the design of recommendation 
    algorithms can substantially influence user engagement patterns and content 
    consumption behaviors, raising important questions about platform 
    responsibility and regulatory oversight.""",

    """Quantum computing represents a paradigm shift in computational capability, 
    leveraging the principles of quantum mechanics to process information in 
    fundamentally different ways compared to classical computing architectures. 
    Quantum bits, or qubits, can exist in superposition states, enabling 
    parallel processing of multiple computational pathways simultaneously. 
    This capability has profound implications for cryptography, drug discovery, 
    materials science, and optimization problems that are currently intractable 
    for classical computers. Leading technology companies and research 
    institutions are investing heavily in quantum hardware development and 
    error correction techniques.""",

    """The global education landscape is undergoing a significant transformation 
    driven by technological innovation and evolving pedagogical approaches. 
    Online learning platforms and digital educational tools have expanded access 
    to educational resources, enabling learners worldwide to engage with 
    high-quality content regardless of geographical constraints. Adaptive 
    learning technologies leverage artificial intelligence to personalize 
    educational experiences, adjusting content difficulty and presentation 
    based on individual learner performance and preferences. These developments 
    have accelerated during recent years, prompting educational institutions 
    to reconsider traditional instructional models and assessment methodologies.""",
]


def run_dr_single(
    text: str,
    chunking_engine: ChunkingEngine,
    shuffle_engine: ShuffleEngine,
    recovery_engine: RecoveryEngine,
    similarity_engine: SimilarityEngine,
    seed: int = 42,
) -> float:
    """Run D&R on a single text and return combined similarity score."""
    cleaned = clean_text(text)
    chunks = chunking_engine.chunk_text(cleaned)

    if not chunks:
        return 0.5

    scores = []
    for i, chunk in enumerate(chunks):
        shuffled = shuffle_engine.disrupt(chunk, seed=seed + i)
        recovered = recovery_engine.recover(shuffled)
        sim = similarity_engine.compute_similarity(chunk, recovered)
        scores.append(sim["combined"])

    return float(np.mean(scores))


def run_benchmark(
    human_texts: List[str],
    ai_texts: List[str],
    threshold: float = 0.85,
) -> dict:
    """
    Run full benchmark on a set of human and AI texts.

    Args:
        human_texts: List of human-written texts.
        ai_texts: List of AI-generated texts.
        threshold: D&R score above this → classify as AI.

    Returns:
        Dictionary with metrics and per-sample results.
    """
    print("\n" + "=" * 60)
    print("  🏋️ SynthDetect — D&R Benchmark")
    print("=" * 60)
    print(f"  Human samples : {len(human_texts)}")
    print(f"  AI samples    : {len(ai_texts)}")
    print(f"  Threshold     : {threshold}")
    print(f"  Model         : {settings.DEFAULT_MODEL}")
    print("=" * 60)

    # Initialize components
    gemini_client = GeminiClient()
    chunking = ChunkingEngine(chunk_size=150, strategy=ChunkStrategy.SEMANTIC)
    shuffling = ShuffleEngine(preserve_ratio=0.2, preserve_boundaries=True)
    recovery = RecoveryEngine(gemini_client=gemini_client, enable_cache=True)
    similarity = SimilarityEngine(alpha=0.6)

    # Ground truth: 0 = human, 1 = AI
    y_true = []
    y_scores = []
    results = []
    total = len(human_texts) + len(ai_texts)

    # Process human texts
    print("\n  Processing human texts...")
    for i, text in enumerate(human_texts):
        word_count = count_words(clean_text(text))
        print(f"    [{i+1}/{len(human_texts)}] Human ({word_count} words)...", end=" ", flush=True)

        start = time.time()
        score = run_dr_single(text, chunking, shuffling, recovery, similarity)
        elapsed = time.time() - start

        y_true.append(0)
        y_scores.append(score)
        results.append({
            "index": i, "label": "human", "score": round(score, 4),
            "words": word_count, "time_s": round(elapsed, 1),
        })
        print(f"score={score:.4f} ({elapsed:.1f}s)")

    # Process AI texts
    print("\n  Processing AI texts...")
    for i, text in enumerate(ai_texts):
        word_count = count_words(clean_text(text))
        print(f"    [{i+1}/{len(ai_texts)}] AI ({word_count} words)...", end=" ", flush=True)

        start = time.time()
        score = run_dr_single(text, chunking, shuffling, recovery, similarity)
        elapsed = time.time() - start

        y_true.append(1)
        y_scores.append(score)
        results.append({
            "index": i, "label": "ai", "score": round(score, 4),
            "words": word_count, "time_s": round(elapsed, 1),
        })
        print(f"score={score:.4f} ({elapsed:.1f}s)")

    # Convert to arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    # Compute metrics
    metrics = compute_binary_metrics(y_true, y_pred, y_scores)

    # Score statistics
    human_scores = y_scores[y_true == 0]
    ai_scores = y_scores[y_true == 1]

    print("\n" + "=" * 60)
    print("  📊 BENCHMARK RESULTS")
    print("=" * 60)

    print(f"""
  ┌───────────────────┬────────────────┐
  │     Metric        │     Value      │
  ├───────────────────┼────────────────┤
  │ AUROC             │  {metrics.get('auroc', 0):.4f}          │
  │ F1 Score          │  {metrics['f1']:.4f}          │
  │ Precision         │  {metrics['precision']:.4f}          │
  │ Recall            │  {metrics['recall']:.4f}          │
  │ Accuracy          │  {metrics['accuracy']:.4f}          │
  └───────────────────┴────────────────┘

  Score Distribution:
  ┌───────────────────┬────────────────┬────────────────┐
  │                   │  👤 Human      │  🤖 AI         │
  ├───────────────────┼────────────────┼────────────────┤
  │ Mean Score        │  {np.mean(human_scores):.4f}          │  {np.mean(ai_scores):.4f}          │
  │ Std Score         │  {np.std(human_scores):.4f}          │  {np.std(ai_scores):.4f}          │
  │ Min Score         │  {np.min(human_scores):.4f}          │  {np.min(ai_scores):.4f}          │
  │ Max Score         │  {np.max(human_scores):.4f}          │  {np.max(ai_scores):.4f}          │
  └───────────────────┴────────────────┴────────────────┘

  Δ (AI - Human) Mean Score: {np.mean(ai_scores) - np.mean(human_scores):+.4f}
    """)

    # LLM usage
    usage = gemini_client.usage_stats()
    print(f"  📈 LLM Usage:")
    print(f"     API calls : {usage['total_calls']}")
    print(f"     Tokens    : {usage['total_input_tokens'] + usage['total_output_tokens']}")
    print(f"     Cost      : ${usage['estimated_cost_usd']:.6f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": settings.DEFAULT_MODEL,
            "threshold": threshold,
            "n_human": len(human_texts),
            "n_ai": len(ai_texts),
        },
        "metrics": metrics,
        "score_stats": {
            "human_mean": round(float(np.mean(human_scores)), 4),
            "human_std": round(float(np.std(human_scores)), 4),
            "ai_mean": round(float(np.mean(ai_scores)), 4),
            "ai_std": round(float(np.std(ai_scores)), 4),
            "delta": round(float(np.mean(ai_scores) - np.mean(human_scores)), 4),
        },
        "per_sample": results,
        "llm_usage": usage,
    }

    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"benchmark_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  💾 Results saved to: {output_path}")
    print("=" * 60 + "\n")

    return output


def main():
    parser = argparse.ArgumentParser(description="SynthDetect D&R Benchmark")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples per class")
    parser.add_argument("--threshold", type=float, default=0.85, help="AI detection threshold")
    args = parser.parse_args()

    n = min(args.samples, len(HUMAN_SAMPLES))
    run_benchmark(
        human_texts=HUMAN_SAMPLES[:n],
        ai_texts=AI_SAMPLES[:n],
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
