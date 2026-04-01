# SynthDetect Methodology

The SynthDetect framework operates on two parallel, orthogonal pipelines to detect AI-generated text.

## 1. Disrupt-and-Recover (D&R) Pipeline
This method exploits the **posterior concentration** phenomenon inherent in all Large Language Models. AI-generated text natively resides in high-probability regions of an LLM's output distribution.

**The Workflow**:
1. **Semantic Chunking**: The input text is split into chunks respecting sentence boundaries using `spaCy`, targeting roughly 200 words per chunk.
2. **Controlled Disruption**: Each chunk's sentences are shuffled randomly, but we preserve the context by rigidly anchoring the first and last sentences in their original positions, and keeping $\sim$20% of the interior sentences static.
3. **LLM Recovery**: A black-box LLM (Gemini 2.5 Flash) is prompted at Temperature 0.0 to unscramble the sentences back into their logical order.
4. **Similarity Scoring**: We score the fidelity of the reconstruction using a hybrid metric:
   - **Semantic ($S_{sem}$)**: Cosine similarity of the original and recovered texts using `all-MiniLM-L6-v2`.
   - **Structural ($S_{str}$)**: A normalized Kendall-$\tau$ rank distance measuring the permutation of sentence indices.
   - **Combined**: $S = 0.6 \cdot S_{sem} + 0.4 \cdot S_{str}$

**The Intuition**: If the original text is AI-generated, the recovering LLM easily "guesses" the highly probable original sequence. If human-written, the LLM will struggle to recover the idiosyncratic original order. Thus, a high similarity score implies AI origin.

## 2. Fine-Grained AI Detection (FAID) Pipeline
While D&R is binary (Human vs. AI), the FAID pipeline attributes the text to a specific model family (e.g., GPT-4, Claude 3, Gemini).

**The Workflow**:
1. **Feature Extraction**: We extract a multi-dimensional vector capturing linguistic properties:
   - Lexical (Type-Token Ratio, Word Length)
   - Syntactic (POS tag distributions via `spaCy`)
   - Semantic (Sentence-level coherence/flow)
   - Stylometric (Readability indices like Flesch-Kincaid)
2. **Contrastive Encoder**: A PyTorch MLP trained with Supervised Contrastive Loss (SupConLoss) maps the raw features into a normalized embedding space. This training forces stylistic fingerprints from the same model to cluster together while pushing different models apart.
3. **Attribution**: We query a FAISS vector database containing previously encoded texts and use distance-weighted k-Nearest Neighbors (k-NN) to infer the top candidate models and their confidence scores.

## 3. Fusion Layer
The final step reconciles both pipelines into a single score:
$$S_{final} = w_{DR} \cdot S_{DR} + (1 - w_{DR}) \cdot S_{FAID}$$

The Fusion Layer also introduces **Collaborative Text Detection**: By analyzing the variance ($\sigma^2$) of D&R similarity scores across individual chunks within a single document, we can identify "cyborg" writing (e.g., a human draft heavily edited sentence-by-sentence by AI). High intra-document variance implies mixed authorship.
