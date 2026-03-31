"""
Multi-Level Feature Extraction for FAID Pipeline

Extracts linguistic features across four dimensions:
- Lexical (10 dims): TTR, word length, rare words, lexical diversity
- Syntactic (variable): dependency relation and POS tag distributions
- Semantic (variable): entity density, coherence metrics
- Stylometric (15 dims): sentence length variance, punctuation, complexity

These features are fed into the contrastive encoder for embedding.

Author: SynthDetect Team
"""

import numpy as np
from typing import Dict, List
from collections import Counter

from src.utils.logger import get_logger

logger = get_logger("faid_pipeline.feature_extraction")


class FeatureExtractor:
    """
    Extracts multi-level linguistic features from text.

    Features are designed to capture stylometric signals that
    distinguish human from AI writing and differentiate model families.
    """

    def __init__(self):
        self._nlp = None
        logger.info("FeatureExtractor initialized")

    @property
    def nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded for feature extraction")
        return self._nlp

    def extract_features(self, text: str) -> Dict[str, np.ndarray]:
        """
        Main feature extraction pipeline.

        Args:
            text: Input text to extract features from.

        Returns:
            Dictionary with feature arrays for each category.
        """
        doc = self.nlp(text)

        features = {
            "lexical": self._extract_lexical_features(doc, text),
            "syntactic": self._extract_syntactic_features(doc),
            "semantic": self._extract_semantic_features(doc),
            "stylometric": self._extract_stylometric_features(doc),
        }

        logger.debug(
            f"Features extracted: "
            + ", ".join(f"{k}={len(v)}" for k, v in features.items())
        )

        return features

    def extract_flat_vector(self, text: str) -> np.ndarray:
        """
        Extract and concatenate all features into a single flat vector.

        Args:
            text: Input text.

        Returns:
            1D numpy array of all features concatenated.
        """
        features = self.extract_features(text)
        return np.concatenate([features[k] for k in sorted(features.keys())])

    def _extract_lexical_features(self, doc, text: str) -> np.ndarray:
        """
        Lexical features (10 dimensions):
        1. Type-Token Ratio (TTR)
        2. Mean word length
        3. Std word length
        4. Hapax legomena ratio (words appearing once)
        5. Long word ratio (>6 chars)
        6. Short word ratio (<4 chars)
        7. Average syllables per word (approximation)
        8. Vocabulary richness (Yule's K)
        9. Function word ratio
        10. Content word ratio
        """
        tokens = [token.text.lower() for token in doc if token.is_alpha]

        if not tokens:
            return np.zeros(10)

        n_tokens = len(tokens)
        n_types = len(set(tokens))
        word_lengths = [len(t) for t in tokens]
        freq_dist = Counter(tokens)

        # 1. TTR
        ttr = n_types / n_tokens

        # 2-3. Word length stats
        mean_word_len = np.mean(word_lengths)
        std_word_len = np.std(word_lengths)

        # 4. Hapax legomena ratio
        hapax = sum(1 for count in freq_dist.values() if count == 1)
        hapax_ratio = hapax / n_tokens

        # 5-6. Word length ratios
        long_ratio = sum(1 for l in word_lengths if l > 6) / n_tokens
        short_ratio = sum(1 for l in word_lengths if l < 4) / n_tokens

        # 7. Syllable approximation (vowel groups)
        def approx_syllables(word):
            vowels = "aeiou"
            count = 0
            prev_vowel = False
            for char in word.lower():
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            return max(1, count)

        avg_syllables = np.mean([approx_syllables(t) for t in tokens])

        # 8. Yule's K (vocabulary richness)
        freq_spectrum = Counter(freq_dist.values())
        M1 = n_tokens
        M2 = sum(freq ** 2 * count for freq, count in freq_spectrum.items())
        yules_k = 10000 * (M2 - M1) / (M1 ** 2) if M1 > 0 else 0

        # 9-10. Function vs content word ratio
        function_pos = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
        n_function = sum(1 for token in doc if token.pos_ in function_pos and token.is_alpha)
        n_content = n_tokens - n_function

        function_ratio = n_function / n_tokens
        content_ratio = n_content / n_tokens

        return np.array([
            ttr, mean_word_len, std_word_len, hapax_ratio,
            long_ratio, short_ratio, avg_syllables, yules_k,
            function_ratio, content_ratio,
        ])

    def _extract_syntactic_features(self, doc) -> np.ndarray:
        """
        Syntactic features (variable dimensions):
        - Dependency relation distribution (normalized counts)
        - POS tag distribution (normalized counts)
        - Parse tree depth statistics
        """
        # Dependency relation distribution
        dep_labels = [token.dep_ for token in doc]
        dep_counter = Counter(dep_labels)
        total_deps = len(dep_labels) or 1

        # Use a fixed set of common dependency labels
        common_deps = [
            "ROOT", "nsubj", "dobj", "iobj", "csubj", "ccomp", "xcomp",
            "nmod", "amod", "advmod", "nummod", "appos", "acl", "advcl",
            "det", "case", "mark", "cc", "conj", "punct", "aux", "cop",
            "compound", "flat", "fixed", "parataxis", "dep",
        ]
        dep_features = np.array([dep_counter.get(d, 0) / total_deps for d in common_deps])

        # POS tag distribution
        pos_tags = [token.pos_ for token in doc]
        pos_counter = Counter(pos_tags)
        total_pos = len(pos_tags) or 1

        common_pos = [
            "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
            "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ",
            "SYM", "VERB", "X",
        ]
        pos_features = np.array([pos_counter.get(p, 0) / total_pos for p in common_pos])

        # Parse tree depth (for each sentence)
        def tree_depth(token):
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
                if depth > 50:
                    break
            return depth

        depths = [tree_depth(token) for token in doc]
        depth_features = np.array([
            np.mean(depths) if depths else 0,
            np.std(depths) if depths else 0,
            np.max(depths) if depths else 0,
        ])

        return np.concatenate([dep_features, pos_features, depth_features])

    def _extract_semantic_features(self, doc) -> np.ndarray:
        """
        Semantic features:
        1. Named entity density
        2. Entity type diversity
        3. Average sentence embedding similarity (coherence)
        4. Unique entity ratio
        """
        n_tokens = len(doc) or 1

        # Entity features
        entities = doc.ents
        n_entities = len(entities)
        entity_density = n_entities / n_tokens

        # Entity type diversity
        entity_types = set(ent.label_ for ent in entities)
        entity_type_diversity = len(entity_types) / max(1, n_entities)

        # Unique entity ratio
        unique_entities = set(ent.text.lower() for ent in entities)
        unique_entity_ratio = len(unique_entities) / max(1, n_entities)

        # Sentence coherence (adjacent sentence similarity via spaCy vectors)
        sentences = list(doc.sents)
        if len(sentences) >= 2:
            coherence_scores = []
            for i in range(len(sentences) - 1):
                s1 = sentences[i].as_doc()
                s2 = sentences[i + 1].as_doc()
                if s1.vector_norm > 0 and s2.vector_norm > 0:
                    coherence_scores.append(s1.similarity(s2))
            avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.5
        else:
            avg_coherence = 0.5

        return np.array([
            entity_density,
            entity_type_diversity,
            unique_entity_ratio,
            avg_coherence,
        ])

    def _extract_stylometric_features(self, doc) -> np.ndarray:
        """
        Stylometric features (15 dimensions):
        1-3. Sentence length (mean, std, max)
        4. Paragraph signal (double newline count / normalized)
        5. Comma density
        6. Semicolon density
        7. Question mark density
        8. Exclamation density
        9. Quote density
        10. Parenthesis density
        11. Sentence complexity (tokens per sentence)
        12. Avg punctuation per sentence
        13. Starts-with-pronoun ratio
        14. Passive voice ratio (approximation)
        15. Conjunction density
        """
        sentences = list(doc.sents)
        n_tokens = len(doc) or 1
        n_sents = len(sentences) or 1

        # Sentence lengths
        sent_lengths = [len(sent) for sent in sentences]
        mean_sent_len = np.mean(sent_lengths) if sent_lengths else 0
        std_sent_len = np.std(sent_lengths) if sent_lengths else 0
        max_sent_len = np.max(sent_lengths) if sent_lengths else 0

        # Paragraph signal (simple heuristic)
        text = doc.text
        paragraph_count = text.count("\n\n") + text.count("\r\n\r\n")
        paragraph_density = paragraph_count / n_sents

        # Punctuation densities
        punct_counts = Counter(token.text for token in doc if token.is_punct)
        comma_density = punct_counts.get(",", 0) / n_tokens
        semicolon_density = punct_counts.get(";", 0) / n_tokens
        question_density = punct_counts.get("?", 0) / n_tokens
        exclamation_density = punct_counts.get("!", 0) / n_tokens
        quote_density = (punct_counts.get('"', 0) + punct_counts.get("'", 0)) / n_tokens
        paren_density = (punct_counts.get("(", 0) + punct_counts.get(")", 0)) / n_tokens

        # Sentence complexity
        avg_tokens_per_sent = n_tokens / n_sents

        # Punctuation per sentence
        total_punct = sum(punct_counts.values())
        punct_per_sent = total_punct / n_sents

        # Starts-with-pronoun ratio
        pronoun_starts = sum(
            1
            for sent in sentences
            if any(token.pos_ == "PRON" for token in sent[:2])
        )
        pronoun_start_ratio = pronoun_starts / n_sents

        # Passive voice approximation (auxpass dependency)
        passive_count = sum(1 for token in doc if token.dep_ in ("auxpass", "nsubjpass"))
        passive_ratio = passive_count / n_tokens

        # Conjunction density
        conj_count = sum(1 for token in doc if token.pos_ in ("CCONJ", "SCONJ"))
        conj_density = conj_count / n_tokens

        return np.array([
            mean_sent_len, std_sent_len, max_sent_len,
            paragraph_density,
            comma_density, semicolon_density, question_density,
            exclamation_density, quote_density, paren_density,
            avg_tokens_per_sent, punct_per_sent,
            pronoun_start_ratio, passive_ratio, conj_density,
        ])
