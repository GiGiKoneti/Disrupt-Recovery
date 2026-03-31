# SynthDetect — AI-Generated Text Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A dual-pipeline framework for detecting AI-generated text via **Disrupt-and-Recover (D&R)** structural analysis and **Fine-Grained AI Detection (FAID)** stylometric attribution.

## ⚡ Quick Start

```bash
# 1. Install dependencies
make setup

# 2. Configure API key
cp .env.example .env
# Edit .env with your Google Gemini API key

# 3. Download datasets
make download-data

# 4. Run tests
make test
```

## 🏗️ Architecture

```
Input Text
    ├── D&R Pipeline: Chunk → Shuffle → Recover (Gemini) → Similarity
    └── FAID Pipeline: Features → Contrastive Encoder → FAISS → Attribution
                    ↓
            Fusion Layer → Classification (AI / Collaborative / Human)
```

## 📁 Project Structure

```
synthdetect/
├── config/          # Configuration (settings, thresholds, model params)
├── src/
│   ├── core/        # Input processing, fusion layer, output
│   ├── dr_pipeline/ # Disrupt-and-Recover detection
│   ├── faid_pipeline/ # Fine-Grained AI Detection
│   ├── models/      # PyTorch contrastive encoder
│   ├── llm_integration/ # Gemini API client + caching
│   └── explainability/  # SHAP analysis, explanations
├── scripts/         # Data prep, training, evaluation
├── notebooks/       # Research Jupyter notebooks
├── tests/           # Unit + integration tests
└── research/        # Paper source, experiment logs
```

## 🔬 Key Features

- **Black-box compatible**: Only needs API access to Gemini (no log-probs)
- **Single LLM call per chunk**: Cost-efficient D&R detection
- **Fine-grained attribution**: Identifies specific model families
- **Collaborative text detection**: Variance-based mixed-authorship signaling
- **Explainable**: Human-readable explanations with SHAP support

## 📊 Target Performance

| Metric | Target | Dataset |
|--------|--------|---------|
| AUROC | > 0.92 | HC3 |
| F1 | > 0.85 | FAIDSet |
| Latency (P95) | < 5s | 1000 words |

## 🛠️ Tech Stack

- **LLM**: Google Gemini 2.5 Flash
- **ML**: PyTorch, scikit-learn, LightGBM
- **NLP**: spaCy, sentence-transformers
- **Vector DB**: FAISS
- **Training**: Kaggle / Google Colab (GPU)

## 📚 Documentation

See [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md) for the complete specification.