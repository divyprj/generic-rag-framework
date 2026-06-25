# Generic RAG Framework

A domain-agnostic, **provider-agnostic** Retrieval-Augmented Generation framework with built-in multi-dimensional evaluation. Ships with **Mars Exploration** and **Quantum Computing** demo datasets. Supports **Groq**, **Gemini**, **OpenAI**, and **Ollama** — switch providers and datasets with zero code changes.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Default_LLM-orange)
![Gemini](https://img.shields.io/badge/Gemini-Supported-green)
![OpenAI](https://img.shields.io/badge/OpenAI-Supported-green)
![Ollama](https://img.shields.io/badge/Ollama-Local-purple)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-blue)
![BGE](https://img.shields.io/badge/BGE--small--en--v1.5-Embeddings-grey)

---

## Quick Start

**1. Configure your API key:**

Create a `.env` file in the project root:

```bash
copy .env.example .env
# Edit .env and add your API key
```

> **Default provider is Groq** (generous free tier). Get a key at [console.groq.com/keys](https://console.groq.com/keys)

**2. Install dependencies:**

```
install.bat
```

**3. Run the application:**

```bash
run.bat                      # default (mars + groq)
run.bat mars groq            # explicit dataset + provider
run.bat quantum gemini       # different dataset + provider
```

**4. Run the evaluation:**

```bash
evaluate.bat                      # default
evaluate.bat mars groq            # explicit
evaluate.bat quantum gemini       # different combo
```

**5. Explore available options:**

```bash
python app.py --list-datasets     # show datasets
python app.py --list-providers    # show provider status
```

That's it — three batch files, any dataset, any provider, no code changes.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Architecture](#project-architecture)
- [LLM Providers](#llm-providers)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Dataset Design](#dataset-design)
- [RAG Pipeline Design](#rag-pipeline-design)
  - [Chunking Strategy](#chunking-strategy)
  - [Embedding Model Choice](#embedding-model-choice)
  - [Retrieval Strategy](#retrieval-strategy)
  - [Prompt Engineering](#prompt-engineering)
- [Evaluation Framework](#evaluation-framework)
  - [Quantitative Metrics](#quantitative-metrics)
  - [Retrieval Performance](#retrieval-performance)
  - [Hallucination Detection](#hallucination-detection)
  - [Human Evaluation Rubric](#human-evaluation-rubric)
- [Sample Outputs](#sample-outputs)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Error Analysis](#error-analysis)
- [Challenges & Lessons Learned](#challenges--lessons-learned)
- [Future Improvements](#future-improvements)

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│               Generic RAG Framework                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌───────────────┐    ┌──────────┐  │
│  │  data/       │───▶│  DatasetConfig │───▶│ Chunking │  │
│  │  <dataset>/  │    │  (dynamic)    │    │ + Embed  │  │
│  └─────────────┘    └───────────────┘    └────┬─────┘  │
│                                                │        │
│                                     ┌──────────▼──────┐ │
│                                     │    ChromaDB     │ │
│                                     │  rag_<dataset>  │ │
│                                     └──────────┬──────┘ │
│                                                │        │
│  ┌─────────────┐    ┌───────────────┐   ┌──────▼──────┐│
│  │   Gemini     │◀──│   Augmented   │◀──│  Top-3      ││
│  │   2.0 Flash  │   │   Prompt      │   │  Retriever  ││
│  └──────┬──────┘    └───────────────┘   └─────────────┘│
│         │                                               │
│  ┌──────▼──────────────────────────────────────────┐   │
│  │  Answer + Source Citations + Confidence Score    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Project Structure

```
EDXSO/
├── install.bat                        # One-click install (Windows)
├── run.bat [dataset]                  # One-click run (Windows)
├── evaluate.bat [dataset]             # One-click evaluation (Windows)
├── smoke_test.py                      # Component verification script
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env.example                       # API key template
├── app.py                             # Interactive CLI application
├── data/
│   ├── mars/                          # Mars Exploration dataset
│   │   ├── dataset.json               # Dataset name + LLM persona
│   │   ├── documents/                 # 8 curated Markdown documents
│   │   │   ├── 01_early_mars_missions.md
│   │   │   ├── 02_spirit_opportunity.md
│   │   │   └── ...
│   │   └── qa_dataset.json            # 15 Q&A pairs with metadata
│   └── quantum/                       # Quantum Computing dataset (Default)
│       ├── dataset.json
│       ├── documents/                 # 8 curated Markdown documents
│       └── qa_dataset.json            # 15 Q&A pairs with metadata
├── src/
│   ├── __init__.py
│   ├── config.py                      # Central config + DatasetConfig
│   ├── document_loader.py             # Document loading & chunking
│   ├── embeddings.py                  # BGE embedding model wrapper
│   ├── vector_store.py                # ChromaDB management
│   ├── retriever.py                   # Top-k retrieval with scoring
│   ├── generator.py                   # Provider-agnostic generation
│   ├── rag_pipeline.py               # End-to-end orchestration
│   └── providers/                     # LLM provider abstraction
│       ├── __init__.py                # Factory + provider registry
│       ├── base.py                    # LLMProvider abstract class
│       ├── gemini_provider.py         # Google Gemini
│       ├── groq_provider.py           # Groq Cloud
│       ├── openai_provider.py         # OpenAI
│       └── ollama_provider.py         # Ollama (local)
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     # Quantitative metrics (5 metrics)
│   ├── retrieval_eval.py              # Retrieval performance (P@k, R@k, MRR)
│   ├── qualitative.py                 # Human evaluation rubric CLI
│   └── run_evaluation.py              # Evaluation runner & dashboard
└── results/                           # Generated evaluation outputs
    ├── evaluation.md                  # Markdown report
    ├── evaluation.csv                 # Flat results table
    └── evaluation.json                # Full structured results
```

---

## LLM Providers

The framework supports four LLM providers. Switch between them with zero code changes.

| Provider | Default Model | API Key Env Var | Free Tier |
|----------|--------------|-----------------|----------|
| **Groq** (default) | llama-3.3-70b-versatile | `GROQ_API_KEY` | ✓ Generous |
| **Gemini** | gemini-2.0-flash | `GOOGLE_API_KEY` | ✓ Limited |
| **OpenAI** | gpt-4o-mini | `OPENAI_API_KEY` | ✗ Paid |
| **Ollama** | llama3 | None (local) | ✓ Free |

### Switching Providers

```bash
# Via CLI
python app.py --provider groq
python app.py --provider gemini
python app.py --provider openai
python app.py --provider ollama

# Via batch scripts
run.bat mars groq
run.bat quantum gemini
```

### Check Provider Status

```bash
python app.py --list-providers
```

Output shows which providers are configured, which keys are missing, and whether Ollama is running.

### Provider Fallback

Enable automatic fallback when the primary provider fails:

```bash
python app.py --provider groq --fallback
```

Fallback order: Groq → Gemini → OpenAI → Ollama. Only providers with valid configuration are attempted. Fallback is **opt-in** (off by default).

### Environment Configuration

```env
# .env file
LLM_PROVIDER=groq          # default provider
GROQ_API_KEY=gsk_...       # Groq key
GOOGLE_API_KEY=AI...       # Gemini key (optional)
OPENAI_API_KEY=sk-...      # OpenAI key (optional)
OLLAMA_BASE_URL=http://localhost:11434  # Ollama (optional)
```

### Using Ollama (Local, No API Key)

1. Install Ollama from [ollama.com](https://ollama.com/download)
2. Pull a model: `ollama pull llama3`
3. Run: `run.bat mars ollama`

## Adding a New Dataset

Add any domain-specific dataset with **zero code changes**:

### Step 1: Create the folder structure

```
data/
└── my_dataset/
    ├── dataset.json
    ├── documents/
    │   ├── 01_topic_one.md
    │   ├── 02_topic_two.md
    │   └── 03_topic_three.md
    └── qa_dataset.json
```

### Step 2: Create `dataset.json`

```json
{
  "name": "My Domain Name",
  "persona": "You are an expert in My Domain with deep knowledge of..."
}
```

### Step 3: Add Markdown documents

Place `.md` files in `documents/`. Recommended: 5-10 documents, 500-1000 words each.

### Step 4: Create `qa_dataset.json`

```json
{
  "questions": [
    {
      "id": "q01",
      "question": "What is...?",
      "expected_answer": "The answer based on your documents.",
      "key_facts": ["fact1", "fact2"],
      "relevant_doc_ids": ["01", "02"],
      "difficulty": "easy",
      "type": "factual"
    }
  ]
}
```

### Step 5: Run it

```bash
run.bat my_dataset          # Interactive mode
evaluate.bat my_dataset     # Evaluation mode
```

The framework automatically:
- Creates a dedicated ChromaDB collection (`rag_my_dataset`)
- Uses the configured persona for LLM prompting
- Ingests documents on first run
- Reuses the index on subsequent runs

---

## Dataset Design

### Domain: History of Quantum Computing

The dataset covers **the complete evolution of quantum information science** from Richard Feynman's initial 1981 proposal through modern hardware achievements, software ecosystems, error correction, post-quantum security, sensing, and chemistry simulations. This domain was chosen because:

- **Factual density**: Rich with specific dates, names, qubit metrics, and scientific formulations.
- **Multi-document relationships**: Hardware platforms (superconducting vs trapped ion) relate directly to error correction surface codes and algorithm constraints (like Shor's algorithm breaking RSA).
- **Temporal ordering**: Clear chronological progression from early mathematical theories in the 1980s to actual physical processors and future post-quantum migration deadlines.
- **Comparison potential**: Diverse competing physical hardware platforms (trapped ions, superconducting qubits, neutral atoms) and algorithms (VQE vs Shor's phase estimation).

### Document Corpus

| # | Document | Topic | Words | Key Entities / Concepts |
|---|----------|-------|-------|-------------------------|
| 01 | Foundations of Quantum Computing | Origins: 1980s-2000s | ~760 | Richard Feynman (1981), David Deutsch (1985), DiVincenzo Criteria (2000) |
| 02 | Quantum Hardware Platforms | Physical implementation technologies | ~820 | Superconducting qubits (IBM, Google Sycamore), Trapped ions, Neutral atoms |
| 03 | Quantum Algorithms and Applications | Primary algorithms and near-term use | ~920 | Shor's algorithm (1994), Grover's search (1996), VQE for chemistry |
| 04 | Quantum Error Correction & Fault Tolerance | QEC architectures and scale milestones | ~770 | Logical vs physical qubits, surface codes, Google's 2023 distance-5 results |
| 05 | Quantum Cryptography and Security | Quantum key distribution & post-quantum standards | ~790 | Post-Quantum Cryptography (PQC), NIST 2024 standards (ML-KEM, ML-DSA), BB84 |
| 06 | Quantum Software and Programming | Development frameworks and compilation | ~760 | Qiskit, Cirq, OpenQASM, Zero-Noise Extrapolation (ZNE), error mitigation |
| 07 | Quantum Sensing and Metrology | Precision metrology and sensors | ~780 | NV Centers in diamond, atomic clocks, SQUIDs, gravimeters |
| 08 | Quantum Materials & Chemistry Sim. | Molecular and physical lattice modeling | ~780 | Fermi-Hubbard model, FeMoco catalyst active space (54 electrons/orbitals), VQE |

**Total corpus size**: ~6,400 words across 8 documents (each 1-2 pages in length).

### Question-Answer Dataset

15 questions spanning 5 types to test different retrieval and reasoning capabilities:

| Type | Count | Difficulty | What It Tests |
|------|-------|------------|---------------|
| **Factual** | 3 | Easy | Single-document fact retrieval |
| **Comparison** | 3 | Medium | Cross-document entity comparison |
| **Multi-document** | 3 | Medium | Information synthesis across docs |
| **Temporal** | 3 | Medium | Chronological reasoning |
| **Reasoning** | 3 | Hard | Inference and causal analysis |

Each question includes:
- `expected_answer`: 2-4 sentence ground truth
- `key_facts`: Specific verifiable facts (names, dates, numbers) for evaluation
- `relevant_doc_ids`: Ground truth for retrieval evaluation
- `difficulty` and `type` labels for stratified analysis

---

## RAG Pipeline Design

### Chunking Strategy

**Method**: `RecursiveCharacterTextSplitter` with markdown-aware separators

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Chunk size | 500 characters | Balances context richness with retrieval precision. Mars documents contain dense factual paragraphs that need enough context to be useful. |
| Chunk overlap | 100 characters | 20% overlap ensures facts at chunk boundaries aren't lost. |
| Separators | `\n## `, `\n### `, `\n\n`, `\n`, `. `, ` ` | Prioritizes splitting at markdown headers and paragraph boundaries, preserving logical structure. |

**Why not smaller chunks?** Testing showed that 300-character chunks often split facts across boundaries (e.g., a date in one chunk, the event in another), degrading retrieval quality.

### Embedding Model Choice

**Model**: `BAAI/bge-small-en-v1.5` (33M parameters, 384 dimensions)

| Criterion | BGE-small | Google text-embedding-004 |
|-----------|-----------|--------------------------|
| Cost | **Free** (local) | API cost per request |
| Latency | **<10ms** (GPU/CPU) | ~200ms (network round-trip) |
| Reproducibility | **Deterministic** | API may change |
| Offline | **Yes** | No |
| Quality (MTEB) | 62.17 | Higher but overkill for 8 docs |
| Quota | **Unlimited** | Rate-limited |

For a small 8-document corpus, the quality difference is negligible, while the operational advantages of local embeddings are significant. Gemini is reserved solely for generation, where its reasoning capabilities matter most.

### Retrieval Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | Cosine similarity | Standard for normalized embeddings; BGE produces L2-normalized vectors |
| Top-k | 3 | Provides sufficient context without overwhelming the prompt; tested against k=5 with similar accuracy but lower precision |
| Min threshold | 0.3 | Filters irrelevant chunks that would add noise to generation |
| Store | ChromaDB (persistent) | Built-in persistence, metadata filtering, lightweight. Better than FAISS for this scale. |

**Confidence Score**: Computed as the mean similarity score of retrieved chunks, providing a human-readable signal of answer reliability.

### Prompt Engineering

The system prompt is designed to maximize groundedness and minimize hallucination:

```
You are a Mars Exploration expert.

Instructions:
- Answer only using the supplied context below.
- If the context does not contain enough information to answer,
  say "The provided documents do not contain enough information
  to fully answer this question."
- Never fabricate or assume facts not present in the context.
- Cite the source document names used in your answer.
- Keep answers concise but complete.
- Use bullet points when listing multiple items.
```

Key design decisions:
1. **Explicit refusal instruction**: Prevents hallucination on out-of-scope questions
2. **Citation requirement**: Forces grounding in retrieved context
3. **Conciseness directive**: Reduces verbose, padded answers that dilute factual density
4. **Bullet point guidance**: Improves readability for comparison and multi-fact answers

---

## Evaluation Framework

The evaluation framework assesses the RAG system across three dimensions: **answer quality** (quantitative), **retrieval quality**, and **human judgment** (qualitative).

### Quantitative Metrics

Five complementary metrics capture different aspects of answer quality:

#### 1. Keyword F1 Score

Measures word-level overlap between generated and expected answers.

$$F_1 = 2 \cdot \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Where precision = (shared words / generated words) and recall = (shared words / expected words). Preprocessing: lowercasing, stop-word removal, punctuation stripping.

**Why F1 over simple overlap?** F1 balances precision (avoiding irrelevant content) with recall (covering key information), penalizing both verbose and incomplete answers.

#### 2. ROUGE-L Score

Longest Common Subsequence (LCS) based metric that captures sentence-level ordering similarity.

$$\text{ROUGE-L} = F_{\text{LCS}} = \frac{(1 + \beta^2) \cdot R_{\text{LCS}} \cdot P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 \cdot P_{\text{LCS}}}$$

Captures whether the generated answer preserves the logical flow of the expected answer, not just keyword presence.

#### 3. Semantic Similarity

Cosine similarity between BGE embeddings of the generated and expected answers.

$$\text{Semantic Sim} = \frac{\mathbf{e}_{\text{gen}} \cdot \mathbf{e}_{\text{exp}}}{\|\mathbf{e}_{\text{gen}}\| \cdot \|\mathbf{e}_{\text{exp}}\|}$$

**Why this matters**: Captures paraphrasing — an answer can be semantically correct without using the same words as the reference.

#### 4. Fact Coverage Score

Percentage of pre-defined `key_facts` found in the generated answer using fuzzy substring matching.

$$\text{Fact Coverage} = \frac{|\text{key\_facts found in answer}|}{|\text{total key\_facts}|}$$

This is the most important metric for RAG: did the system retrieve and include the specific facts the question requires?

#### 5. Hallucination Rate

Estimates the proportion of generated sentences not supported by retrieved context.

For each sentence in the generated answer:
1. Tokenize into content words (remove stop words)
2. Check word overlap with each retrieved context chunk
3. Flag as "potentially hallucinated" if <30% overlap with best-matching chunk

$$\text{Hallucination Rate} = \frac{|\text{unsupported sentences}|}{|\text{total sentences}|}$$

**Note**: This is a heuristic estimate. A sentence about "Mars" will overlap with any context chunk; the threshold is tuned to catch fabricated details (dates, names, measurements) rather than general statements.

### Composite RAG Score

Weighted combination of all metrics into a single 0-1 score:

| Metric | Weight | Rationale |
|--------|--------|-----------|
| Fact Coverage | 30% | Core purpose of RAG: retrieve specific facts |
| Semantic Similarity | 25% | Overall meaning alignment |
| ROUGE-L | 20% | Structural and ordering similarity |
| Keyword F1 | 15% | Lexical precision and recall |
| (1 - Hallucination) | 10% | Penalizes fabricated content |

### Retrieval Performance

Three metrics evaluate whether the retriever finds the right document chunks:

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **Precision@k** | relevant_retrieved / k | % of retrieved chunks from correct documents |
| **Recall@k** | relevant_retrieved_docs / total_relevant_docs | % of relevant documents represented |
| **MRR** | 1 / rank_of_first_relevant | How quickly the first relevant chunk appears |

Ground truth: Each question in `qa_dataset.json` specifies `relevant_doc_ids` — the documents that contain the answer.

### Human Evaluation Rubric

An interactive CLI interface for qualitative assessment:

| Dimension | Scale | Anchor Points |
|-----------|-------|---------------|
| **Coherence** | 1-5 | 1=Incoherent/contradictory, 3=Understandable but awkward, 5=Clear and well-structured |
| **Completeness** | 1-5 | 1=Missing all key facts, 3=Covers some facts, 5=Comprehensive coverage |
| **Factual Accuracy** | 1-5 | 1=Multiple factual errors, 3=Mostly correct with minor errors, 5=All facts verified correct |
| **Groundedness** | 1-5 | 1=Mostly unsupported claims, 3=Partially grounded, 5=Every claim traceable to context |

Features:
- Side-by-side display of expected vs. generated answers
- Retrieved context shown for groundedness verification
- Optional free-text notes per question
- Progress saving (resume interrupted evaluations)
- Aggregate statistics on completion

---

## Sample Outputs

### Interactive Query

```
🔴 Mars Exploration RAG System 🔴
   Powered by BGE Embeddings + Gemini 2.0 Flash

❯ What was the first helicopter to fly on Mars?

──────────────── Answer ─────────────────

Ingenuity was the first helicopter to achieve powered flight on another
planet. It flew on Mars on April 19, 2021, as part of NASA's Mars 2020
mission alongside the Perseverance rover.

Key specifications:
• Mass: 1.8 kg
• Rotor span: 1.2 meters
• Total flights: 72
• Deployment site: Jezero Crater

Sources: 04_perseverance_ingenuity.md

  🟢 Confidence: 91%  |  ⏱️  Latency: 1247ms

──────────────── Sources ────────────────
┌───┬─────────────────────────────────┬───────┬──────────────┐
│ # │ Document                        │ Chunk │ Similarity   │
├───┼─────────────────────────────────┼───────┼──────────────┤
│ 1 │ 04_perseverance_ingenuity.md    │ 3     │ 0.921 █████▓ │
│ 2 │ 04_perseverance_ingenuity.md    │ 2     │ 0.874 █████▒ │
│ 3 │ 08_future_missions.md           │ 1     │ 0.612 ███▒░░ │
└───┴─────────────────────────────────┴───────┴──────────────┘
```

### Evaluation Report (excerpt)

```
┌─────┬──────────────────────────────────┬──────┬────────┬──────┬──────┬──────┬───────┐
│ ID  │ Question                         │ KW   │ ROUGE  │ Sem  │ Fact │ Hall │ RAG   │
│     │                                  │ F1   │ -L     │ Sim  │ Cov  │ Rate │ Score │
├─────┼──────────────────────────────────┼──────┼────────┼──────┼──────┼──────┼───────┤
│ q01 │ First spacecraft to fly by Mars? │ 0.82 │ 0.75   │ 0.91 │ 1.00 │ 0.00 │ 0.88  │
│ q02 │ Compare Spirit and Opportunity   │ 0.65 │ 0.58   │ 0.84 │ 0.75 │ 0.10 │ 0.72  │
│ ... │ ...                              │ ...  │ ...    │ ...  │ ...  │ ...  │ ...   │
├─────┼──────────────────────────────────┼──────┼────────┼──────┼──────┼──────┼───────┤
│     │ AVERAGE                          │ 0.73 │ 0.66   │ 0.87 │ 0.83 │ 0.08 │ 0.79  │
└─────┴──────────────────────────────────┴──────┴────────┴──────┴──────┴──────┴───────┘
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Quick Setup (Windows — Recommended)

```
1. Create .env file with: GOOGLE_API_KEY=your_key_here
2. Double-click install.bat
3. Double-click run.bat
```

See [Quick Start](#quick-start) at the top of this README.

### Manual Setup

```bash
# Clone the repository
git clone https://github.com/your-username/mars-rag-system.git
cd mars-rag-system

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API key
copy .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### Automation Scripts (Windows)

| Script | Purpose |
|--------|---------|
| `install.bat` | Creates venv, installs dependencies, verifies modules |
| `run.bat` | Validates setup, checks API key, launches interactive CLI |
| `evaluate.bat` | Validates setup, runs full evaluation, shows report locations |

All scripts use paths relative to the project root and work from any directory.

### Usage

#### Interactive Mode
```bash
python app.py
```

#### Single Query
```bash
python app.py --query "What did Curiosity discover about methane on Mars?"
```

#### Force Re-ingestion
```bash
python app.py --ingest
```

#### Run Evaluation
```bash
# Full quantitative evaluation
python -m evaluation.run_evaluation

# Include human evaluation
python -m evaluation.run_evaluation --human

# Evaluate specific number of questions
python -m evaluation.run_evaluation --questions 5
```

---

## Error Analysis

### Expected Strengths
- **Factual questions**: High accuracy for single-document lookups with specific dates and names
- **Temporal questions**: Good performance when chronological information is within a single chunk
- **Source citation**: Prompt engineering ensures grounded answers

### Expected Weaknesses
- **Multi-document synthesis**: May retrieve chunks from only one relevant document, missing information from others
- **Comparison questions**: Requires information from 2+ documents; retrieval may favor one over another
- **Numerical precision**: Small chunk sizes may split tables or lists of specifications

### Mitigation Strategies
1. **Chunk overlap (100 chars)**: Ensures facts at boundaries are duplicated
2. **Top-k=3 retrieval**: Increases chance of capturing multiple relevant documents
3. **Markdown-aware splitting**: Preserves header context within chunks

---

## Challenges & Lessons Learned

### 1. Embedding Model Selection
**Challenge**: Balancing embedding quality with operational simplicity.
**Resolution**: Local BGE model eliminates API costs and latency while providing sufficient quality for a small corpus. The quality gap between BGE-small and larger models is negligible at this scale.

### 2. Chunk Size Optimization
**Challenge**: Finding the right balance between retrieval precision and context completeness.
**Resolution**: 500-character chunks with markdown-aware splitting preserve logical sections. Too small (200) fragments facts; too large (1000) reduces retrieval precision.

### 3. Hallucination Prevention
**Challenge**: LLMs tend to generate plausible but unsupported facts about well-known topics.
**Resolution**: Strict prompt instructions ("Answer ONLY from context") combined with a hallucination detection metric in evaluation. Temperature set to 0.1 to minimize creative generation.

### 4. Evaluation Design
**Challenge**: No single metric captures answer quality comprehensively.
**Resolution**: Multi-dimensional evaluation with five quantitative metrics, three retrieval metrics, and four qualitative dimensions. The composite RAG Score provides a quick summary while individual metrics enable diagnosis.

### 5. Cross-Document Retrieval
**Challenge**: Questions requiring synthesis across documents may only retrieve chunks from one source.
**Resolution**: Top-k=3 with diversity in document sources helps, but remains a fundamental limitation of naive vector similarity retrieval.

---

## Future Improvements

1. **Hybrid Retrieval**: Combine BM25 keyword search with vector similarity for better handling of specific names and dates
2. **Cross-Encoder Reranking**: Add a second-stage reranker (e.g., `ms-marco-MiniLM-L-6-v2`) to improve precision
3. **Conversation Memory**: Support follow-up questions with context from previous queries
4. **Streaming Responses**: Stream Gemini output for better perceived latency
5. **Web Interface**: Gradio or Streamlit frontend for non-technical users
6. **Automated Hallucination Detection**: Use an LLM-as-judge approach for more nuanced hallucination scoring
7. **Document Expansion**: Add more documents on Chinese and Indian Mars programs

---

## License

This project is created for educational/evaluation purposes. The dataset content is based on publicly available information about NASA, ESA, and other space agencies' Mars missions.
