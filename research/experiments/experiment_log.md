# SynthDetect Experiment Log

## Experiment 1: D&R Pipeline Prototype Validation
**Date**: 2026-04-01
**Objective**: Validate the core hypothesis that AI-generated text is recovered with higher fidelity by an LLM after disruption (posterior concentration theory).
**Model**: Google Gemini 2.5 Flash
**Settings**:
- Chunking: Semantic (150 words)
- Shuffling: Sentence-level, 20% preserved boundaries
- Similarity: Hybrid ($\alpha=0.6$) using all-MiniLM-L6-v2 and Kendall-$\tau$

**Results**:
- Human Mean Combined Sim: 0.9199 ($\sigma = 0.0801$)
- AI Mean Combined Sim: 0.9706 ($\sigma = 0.0294$)
- **$\Delta$ Combined**: +0.0507 (Hypothesis Supported)
- **$\Delta$ Structural**: +0.1243 (Strongest discriminative signal)

**Conclusion**: The hypothesis holds. The LLM reconstructed the AI-generated sentence ordering remarkably well compared to human text.

---

## Experiment 2: Benchmark Evaluation (5 vs 5)
**Date**: 2026-04-01
**Objective**: Evaluate D&R performance on a small sample of 5 human vs 5 AI texts.
**Datasets**: Built-in 100-word samples
**Model**: Google Gemini 2.5 Flash

**Results**:
- AUROC: 0.5400
- F1 Score: 0.6667
- Precision: 0.5000 | Recall: 1.0000 | Accuracy: 0.5000
- Human Mean Score: 0.9688
- AI Mean Score: 0.9720
- **$\Delta$ Mean Score**: +0.0032

**Observations**:
- The detection signal ($\Delta$) dropped significantly (+0.0507 $\rightarrow$ +0.0032) when applying the pipeline to shorter texts (~100 words).
- Many chunks were too small for meaningful permutations, leading to 0 disruption and artificial 1.0 similarity scores.

**Next Action Items**:
- Benchmark on official datasets (HC3, M4GT) with long documents (>200 words).
- Tune the AI detection classification threshold (currently 0.85) via ROC analysis.
- Increase chunk size minimums or reduce overlap for short pieces.
