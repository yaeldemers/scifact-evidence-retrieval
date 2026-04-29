# Scientific Evidence Retrieval Lab
An experimental ML systems project exploring semantic retrieval, retrieval-augmented generation (RAG), and embedding-space interpretability using scientific literature datasets.
The system combines dense vector search, evidence-grounded generation, retrieval evaluation, and embedding analysis to investigate how modern retrieval pipelines behave under real-world information retrieval conditions.

## Overview
This project implements a lightweight scientific evidence retrieval and fact-checking pipeline built on top of the SciFact dataset from the BEIR benchmark suite.
Given a scientific claim or question, the system:
- converts the query into a dense vector embedding
- retrieves semantically relevant scientific abstracts using vector similarity search
- reranks candidates using a cross-encoder for more precise ranking
- optionally generates a grounded response using retrieved evidence
- evaluates retrieval and generation quality
- analyzes retrieval behavior through embedding-space inspection and failure diagnostics

The project focuses on retrieval reliability, grounded generation, and interpretable analysis of embedding behavior.

## Dataset
The project uses the SciFact dataset from the BEIR (Benchmarking Information Retrieval) benchmark collection.
SciFact contains:
- 5,183 scientific paper abstracts
- 300 test queries (scientific claims)
- Expert-annotated relevance mappings between claims and supporting evidence

This enables reproducible evaluation of:
- semantic retrieval systems
- dense vector search pipelines
- retrieval-augmented generation workflows
- retrieval failure modes

## System Architecture
```text
User Query / Scientific Claim
              ↓
      Bi-Encoder Embedding
    (all-MiniLM-L6-v2)
              ↓
    Vector Retrieval (ChromaDB)
      top-50 candidates
              ↓
    Cross-Encoder Reranking
    (ms-marco-MiniLM-L-6-v2)
              ↓
      Top-10 Relevant Papers
              ↓
        Context Assembly
              ↓
      Retrieval-Augmented
         Generation
              ↓
      Evidence-Grounded
            Output
```

## Features

### Part 1 — Retrieval Pipeline (complete)
- Dense vector search over all 5,183 scientific abstracts
- Persistent ChromaDB index (skip re-encoding on subsequent runs)
- BM25 lexical search baseline
- Hybrid search via Reciprocal Rank Fusion (BM25 + dense)
- Cross-encoder reranking stage
- Evaluation framework: Recall@k, nDCG@k, MRR@k
- Pre-computed reranking caches (dense and hybrid) for fast re-evaluation

### Part 2 — Retrieval-Augmented Generation (planned)
- Evidence-grounded answer generation
- Context injection from retrieved scientific papers
- Citation-aware response generation
- Hallucination and grounding analysis

### Part 3 — Embedding Interpretability (planned)
- Embedding-space visualization
- Cluster analysis using dimensionality reduction
- Successful vs failed retrieval comparison
- Embedding distance analysis
- Lightweight probing experiments

## Part 1 Results (SciFact, k=10)

| Method | Recall@10 | nDCG@10 | MRR@10 |
|--------|-----------|---------|--------|
| Dense (all-MiniLM-L6-v2) | 0.793 | 0.645 | 0.605 |
| BM25 | 0.703 | 0.560 | 0.524 |
| Dense + Reranking | **0.837** | **0.694** | **0.663** |
| Hybrid RRF (no rerank) | 0.803 | 0.646 | 0.607 |
| Hybrid RRF + Reranking | 0.830 | 0.692 | 0.660 |

**Best pipeline: Dense + Cross-Encoder Reranking.** On SciFact, dense retrieval already captures most relevant candidates in its top-50, so BM25 fusion (RRF) adds noise rather than signal before reranking. The cross-encoder reranker provides the largest quality jump across all metrics.

### Example Workflow
Input claim:
```text
"Vitamin D reduces severity of respiratory infections"
```
Pipeline behavior:
```text
Claim
  ↓
Bi-encoder embedding
  ↓
Top-50 candidate retrieval (ChromaDB)
  ↓
Cross-encoder reranking
  ↓
Top-10 relevant scientific abstracts
  ↓
RAG-based synthesis (planned)
  ↓
Evidence-grounded response (planned)
```

### Example output (planned):
```text
Retrieved Evidence:
- Paper A reports reduced severity correlation
- Paper B reports insufficient evidence

Generated Response:
Current literature suggests a possible relationship
between vitamin D levels and respiratory infection outcomes,
though findings remain mixed across studies.
```

### Experimental Focus Areas
- Dense retrieval behavior
- Retrieval ranking quality
- Retrieval vs generation failure attribution
- Context quality effects on generation
- Embedding-space structure and clustering
- Semantic ambiguity in vector search systems
- Grounded response generation reliability

## Repository Structure
```text
scientific-evidence-retrieval/

  notebooks/
    01_semantic_search.ipynb        # Part 1: retrieval pipeline and evaluation (complete)
    02_rag_failure_diagnosis.ipynb  # Part 2: RAG + failure analysis (planned)
    data/
      chroma/                       # Persistent ChromaDB vector index (5,183 docs)
      reranked_cache.json           # Pre-computed dense + reranker results (300 queries)
      hybrid_reranked_cache.json    # Pre-computed RRF + reranker results (300 queries)

  src/                              # Planned: shared pipeline modules
    embeddings.py
    retrieval.py
    rag.py
    evaluation.py
    analysis.py
    visualization.py

  outputs/                          # Planned: saved figures, reports, experiment runs
    indexes/
    figures/
    reports/
    experiment_runs/
```

## Technical Components
- `sentence-transformers/all-MiniLM-L6-v2` — bi-encoder for dense retrieval
- `cross-encoder/ms-marco-MiniLM-L-6-v2` — cross-encoder reranker
- ChromaDB — persistent vector database
- rank-bm25 (BM25Okapi) — lexical search baseline
- Reciprocal Rank Fusion (RRF) — hybrid retrieval fusion
- BEIR / SciFact — dataset and evaluation framework

## Project Objectives
- Evaluate semantic retrieval quality on scientific literature
- Investigate retrieval-augmented generation reliability
- Analyze retrieval and grounding failure modes
- Study geometric structure of embedding representations
- Explore interpretable analysis techniques for retrieval systems
