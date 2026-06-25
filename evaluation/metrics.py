"""Quantitative evaluation metrics for RAG system answers.

Provides:
- Keyword F1: word-overlap F1 between generated and expected answers
- ROUGE-L: longest common subsequence overlap (F-measure)
- Semantic Similarity: cosine similarity between embedding vectors
- Fact Coverage: fuzzy match of pre-defined key facts in generated answer
- Hallucination Rate: sentences in generated answer unsupported by context
- Composite RAG Score: weighted average of all metrics
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable

import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity

# ── helpers ──────────────────────────────────────────────────────────────────

_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "against",
    "this", "that", "these", "those", "it", "its", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "whom",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords, return token list."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t and t not in _STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on common delimiters."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ── public metrics ───────────────────────────────────────────────────────────


def keyword_f1(generated: str, expected: str) -> float:
    """F1 score based on word-overlap between generated and expected answers.

    Both strings are lowercased, punctuation is stripped, and English
    stopwords are removed before computing precision, recall, and F1.

    Args:
        generated: The model-generated answer.
        expected:  The gold / reference answer.

    Returns:
        F1 score in [0.0, 1.0].
    """
    gen_tokens = set(_tokenize(generated))
    exp_tokens = set(_tokenize(expected))

    if not gen_tokens or not exp_tokens:
        return 0.0

    common = gen_tokens & exp_tokens
    precision = len(common) / len(gen_tokens) if gen_tokens else 0.0
    recall = len(common) / len(exp_tokens) if exp_tokens else 0.0

    if precision + recall == 0.0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def rouge_l_score(generated: str, expected: str) -> float:
    """ROUGE-L F-measure using longest common subsequence overlap.

    Uses the ``rouge_score`` library's ``RougeScorer`` with
    ``use_stemmer=True`` for robustness to morphological variation.

    Args:
        generated: The model-generated answer.
        expected:  The gold / reference answer.

    Returns:
        ROUGE-L F-measure in [0.0, 1.0].
    """
    if not generated.strip() or not expected.strip():
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(expected, generated)
    return float(scores["rougeL"].fmeasure)


def semantic_similarity(
    generated: str,
    expected: str,
    embed_fn: Callable[[str], list[float]],
) -> float:
    """Cosine similarity between embedding vectors of the two texts.

    Args:
        generated: The model-generated answer.
        expected:  The gold / reference answer.
        embed_fn:  Callable that maps a string to a list of floats (embedding).

    Returns:
        Cosine similarity in [0.0, 1.0] (clamped to non-negative).
    """
    if not generated.strip() or not expected.strip():
        return 0.0

    gen_emb = np.array(embed_fn(generated)).reshape(1, -1)
    exp_emb = np.array(embed_fn(expected)).reshape(1, -1)
    sim = cosine_similarity(gen_emb, exp_emb)[0][0]
    return float(max(0.0, min(1.0, sim)))


def fact_coverage(generated: str, key_facts: list[str]) -> float:
    """Percentage of *key_facts* found in the generated answer.

    Matching is case-insensitive and tolerates minor surrounding
    context (e.g. ``"1965"`` matches ``"in 1965"``).  Each fact is
    treated as a substring search after lowercasing.

    Args:
        generated: The model-generated answer.
        key_facts: List of short factual strings that should appear.

    Returns:
        Fraction of facts matched in [0.0, 1.0].
    """
    if not key_facts:
        return 1.0  # vacuously true — no facts required

    generated_lower = generated.lower()
    matched = 0
    for fact in key_facts:
        fact_lower = fact.lower().strip()
        if not fact_lower:
            matched += 1  # empty fact counts as matched
            continue
        # Direct substring match
        if fact_lower in generated_lower:
            matched += 1
            continue
        # Token-level fuzzy: all content words of the fact appear in the answer
        fact_tokens = _tokenize(fact)
        if fact_tokens and all(tok in generated_lower for tok in fact_tokens):
            matched += 1

    return matched / len(key_facts)


def hallucination_score(
    generated: str,
    context_chunks: list[str],
) -> float:
    """Estimate the fraction of generated sentences NOT supported by context.

    A sentence is considered *unsupported* (hallucinated) if fewer than
    30 % of its content words appear in **any** context chunk.

    Args:
        generated:      The model-generated answer.
        context_chunks: Retrieved text passages used as grounding context.

    Returns:
        Hallucination rate in [0.0, 1.0].
        0.0 = fully grounded, 1.0 = entirely hallucinated.
    """
    sentences = _split_sentences(generated)
    if not sentences:
        return 0.0

    # Pre-compute set of content words across all context chunks
    context_words_per_chunk: list[set[str]] = [
        set(_tokenize(chunk)) for chunk in context_chunks
    ]

    hallucinated = 0
    for sentence in sentences:
        sent_tokens = set(_tokenize(sentence))
        if not sent_tokens:
            continue  # skip empty / pure-stopword sentences

        # Check against each chunk — sentence is grounded if *any* chunk
        # covers ≥30 % of its content words.
        grounded = False
        for chunk_words in context_words_per_chunk:
            overlap = sent_tokens & chunk_words
            if len(overlap) / len(sent_tokens) >= 0.30:
                grounded = True
                break

        if not grounded:
            hallucinated += 1

    total_non_empty = sum(
        1 for s in sentences if _tokenize(s)
    )
    if total_non_empty == 0:
        return 0.0

    return hallucinated / total_non_empty


def composite_rag_score(
    keyword_f1_val: float,
    rouge_l_val: float,
    semantic_sim_val: float,
    fact_coverage_val: float,
    hallucination_val: float,
) -> float:
    """Weighted composite RAG quality score.

    Weights:
        - Fact Coverage   : 30 %
        - Semantic Sim    : 25 %
        - ROUGE-L         : 20 %
        - Keyword F1      : 15 %
        - (1 - Hallucination) : 10 %

    Args:
        keyword_f1_val:    Keyword F1 score [0, 1].
        rouge_l_val:       ROUGE-L F-measure [0, 1].
        semantic_sim_val:  Semantic similarity [0, 1].
        fact_coverage_val: Fact coverage [0, 1].
        hallucination_val: Hallucination rate [0, 1].

    Returns:
        Composite score in [0.0, 1.0].
    """
    return (
        0.30 * fact_coverage_val
        + 0.25 * semantic_sim_val
        + 0.20 * rouge_l_val
        + 0.15 * keyword_f1_val
        + 0.10 * (1.0 - hallucination_val)
    )


def compute_all_metrics(
    generated: str,
    expected: str,
    key_facts: list[str],
    context_chunks: list[str],
    embed_fn: Callable[[str], list[float]],
) -> dict[str, float]:
    """Compute every metric and return a flat dictionary.

    Args:
        generated:      Model-generated answer.
        expected:       Gold / reference answer.
        key_facts:      List of factual strings expected in the answer.
        context_chunks: Retrieved context passages.
        embed_fn:       Embedding function ``str → list[float]``.

    Returns:
        Dictionary with keys:
        ``keyword_f1``, ``rouge_l``, ``semantic_similarity``,
        ``fact_coverage``, ``hallucination_rate``, ``composite_score``.
    """
    kw_f1 = keyword_f1(generated, expected)
    rl = rouge_l_score(generated, expected)
    sem_sim = semantic_similarity(generated, expected, embed_fn)
    fc = fact_coverage(generated, key_facts)
    hall = hallucination_score(generated, context_chunks)
    comp = composite_rag_score(kw_f1, rl, sem_sim, fc, hall)

    return {
        "keyword_f1": round(kw_f1, 4),
        "rouge_l": round(rl, 4),
        "semantic_similarity": round(sem_sim, 4),
        "fact_coverage": round(fc, 4),
        "hallucination_rate": round(hall, 4),
        "composite_score": round(comp, 4),
    }
