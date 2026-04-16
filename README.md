# Lebanese Legal RAG Assistant

<p align="center">
  <img src="frontend/public/logo.png" alt="Lebanese Legal RAG Assistant Logo" width="320" />
</p>

Production-minded Arabic legal assistant for Lebanese law, built with a modern RAG stack:
- FastAPI backend
- React + Vite frontend
- Arabic embedding retrieval
- Optional hybrid search + reranking
- Grounded answer verification with citation-first behavior

This project is designed to demonstrate strong engineering for high-stakes QA: retrieval quality, abstention safety, and measurable evaluation.

## Production Web Features Added

- Account settings modal (profile, active sessions, compliance status)
- Password change endpoint + UI with strong password policy
- Account deletion endpoint + UI (password + explicit `DELETE` confirmation)
- Conversation deletion controls in sidebar
- Consent gating before login/register:
  - legal disclaimer acceptance
  - privacy/data-processing acceptance
  - essential cookies/local storage acceptance
- Security response headers middleware
- CORS allowlist via environment variable

## Why This Project Stands Out

Most chatbot demos optimize for fluent output.
This one optimizes for legal reliability:

- Retrieves from a local legal corpus (not open-ended generation).
- Forces citation-linked answers.
- Adds verifier and self-check layers to reduce overconfident mistakes.
- Supports insufficient-basis fallback when evidence is weak.
- Includes benchmark tooling to compare baseline vs enhanced pipelines.

## What It Does

For each user message:

1. Detects intent:
- Legal question -> legal pipeline
- Non-legal question -> normal conversational assistant mode

2. Legal pipeline:
- Arabic query normalization
- Query paraphrasing
- Retrieval (`dense` or `hybrid`)
- Optional reranking
- Evidence verification
- Answer generation in simple language for non-lawyers
- Post-generation groundedness self-check
- Citation summary + source cards

3. If legal evidence is insufficient:
- Returns a safe, explicit insufficient-basis response
- Asks for clarifying context when needed

## Tech Stack

Backend:
- `FastAPI`
- `sentence-transformers` (Arabic embeddings)
- `torch`
- `rank-bm25` (hybrid retrieval option)
- `OpenAI API` (intent, verification, legal synthesis, self-check)

Frontend:
- `React`
- `TypeScript`
- `Vite`
- `TailwindCSS`

## Current Retrieval and Reasoning Modes

### Baseline
- Dense retrieval using Arabic SBERT embeddings
- LLM-based evidence verification
- Structured legal answer with citations

### Enhanced (SOTA-style architecture, optional)
- Hybrid retrieval: dense + BM25 fusion
- Cross-encoder reranker (feature-flagged)
- Confidence-aware abstention
- Post-generation self-check against retrieved evidence

All enhancements are additive and configurable; baseline remains available.

## Repository Structure

```text
LegalChatbot/
|- Backend/
|  |- app.py
|  |- semantic_search.py
|  |- requirements.txt
|  |- scripts/
|  |  |- precompute_embeddings.py
|  |  |- eval_pipeline.py
|  |  |- build_eval_v1.py
|  |  `- run_benchmark_matrix.py
|  `- data/
|     |- cleaned/
|     `- eval/
|        |- legal_eval_sample.jsonl
|        `- legal_eval_v1.jsonl
|- frontend/
|  |- src/
|  `- package.json
`- README.md
```

## Setup

### 1) Backend

```bash
cd Backend
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create `Backend/.env`:

```env
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
ALLOWED_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
POLICY_VERSION=2026-04-16

PRELOAD_EMBEDDINGS=1
EMBEDDINGS_REQUIRE_PREBUILT=1

PARAPHRASE_COUNT=12
MAX_EVIDENCE_CANDIDATES=20
MAX_FINAL_EVIDENCE=6

RETRIEVAL_MODE=dense
DENSE_WEIGHT=0.75
BM25_WEIGHT=0.25
HYBRID_PER_QUERY_K=40

RERANKER_ENABLED=0
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANKER_TOP_N=30
RERANKER_WEIGHT=0.35

MIN_ANSWER_CONFIDENCE=0.35
SELF_CHECK_ENABLED=1
```

Run API:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### 2) Frontend

```bash
cd frontend
npm install
```

Create `frontend/.env`:

```env
VITE_API_BASE=http://127.0.0.1:8000
```

Run UI:

```bash
npm run dev
```

## Fast Startup in Deployment

To avoid slow first requests, precompute embeddings and reuse cache:

```bash
cd Backend
python scripts/precompute_embeddings.py
```

Recommended deploy env:

```env
PRELOAD_EMBEDDINGS=1
EMBEDDINGS_REQUIRE_PREBUILT=1
# optional persistent path:
# EMBEDDINGS_PATH=/persistent/path/corpus_emb.pt
```

## API

### `POST /search`

Request:

```json
{
  "query": "What is the penalty for theft in Lebanese law?",
  "top_k": 40
}
```

Response:
- `answer`: structured legal or conversational response
- `sources`: source cards with citation tags
- `raw_sources`: retrieved legal rows used by the pipeline

## Benchmarking

### Build 120-query evaluation set

```bash
python Backend/scripts/build_eval_v1.py
```

Dataset composition:
- 60 covered
- 30 edge/ambiguous
- 30 uncovered

### Run single evaluation

```bash
python Backend/scripts/eval_pipeline.py \
  --dataset Backend/data/eval/legal_eval_v1.jsonl \
  --api-base http://127.0.0.1:8000 \
  --top-k 40 \
  --out-json Backend/data/eval/results_v1/run.json
```

### Run benchmark matrix

```bash
python Backend/scripts/run_benchmark_matrix.py \
  --dataset Backend/data/eval/legal_eval_v1.jsonl \
  --api-base http://127.0.0.1:8000 \
  --top-k 40 \
  --out-dir Backend/data/eval/results_v1
```

Matrix compares:
- `baseline_dense`
- `enhanced_hybrid_conf_030`
- `enhanced_hybrid_conf_035`
- `enhanced_hybrid_conf_040`

## Observed Preliminary Result (Smoke Slice)

On a 20-query covered subset:
- Baseline insufficient-basis rate: `30%`
- Enhanced (hybrid + reranker, conf=0.30) insufficient-basis rate: `15%`
- Citation hit (answer and source cards): `90%` in both runs
- Failures: `0`

This indicates improved answer availability with preserved citation grounding on that slice.

## Product and Safety Design Choices

- Citation-first legal QA, not free-form legal speculation
- Abstain when evidence is weak
- Separate legal and non-legal conversation paths
- Groundedness self-check before final output
- Evaluation scripts included to support measurable iteration


## License

MIT
