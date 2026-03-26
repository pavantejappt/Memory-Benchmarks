# LoComo Memory Benchmark

Benchmarks various memory backends on the [LoComo dataset](https://github.com/snap-research/locomo/tree/main) — 10 long conversations with ~200 Q&A pairs each across 4 categories: single-hop, temporal, multi-hop, and open-domain.

---

## Techniques

| Technique | Type | Backend | Notes |
|---|---|---|---|
| `mem0_local` | Memory | Qdrant + local vLLM | OSS mem0, fully local |
| `langmem` | Memory | InMemoryStore + local vLLM | RAM only, no persistence |
| `rag` | RAG | Qdrant + local vLLM | Fixed/turn/multi-turn chunking, dense/sparse/hybrid search |

---

## Results

LLM Judge Score (higher is better).

All runs use **Qwen3-32B** (vLLM, local) as both the RAG/memory model and judge, with **BGE-base-en-v1.5** embeddings.

| Method | Search | Chunking | Thinking | Single-Hop(%) | Multi-Hop(%) | Open Domain(%) | Temporal(%) | Overall(%) |
|---|---|---|---|---|---|---|---|---|
| RAG | dense | fixed | off | 46.5 | 58.3 | 67.7 | 49.8 | 59.5 |
| RAG | dense | fixed | on | 50.7 | 50.0 | 70.4 | 53.0 | 61.9 |
| RAG | sparse (BM25) | fixed | off | 40.4 | 54.2 | 76.1 | 47.7 | 62.3 |
| RAG | hybrid | fixed | off | - | - | - | - | in progress |
| RAG | dense | turn-level | off | - | - | - | - | in progress |
| RAG | dense | multi-turn (3) | off | - | - | - | - | in progress |
| RAG | dense | full context | off | - | - | - | - | in progress |
| mem0_local | - | - | off | 40.4 | 42.7 | 59.7 | 49.8 | 53.1 |
| mem0_local | - | - | on | - | - | - | - | in progress |
| LangMem | - | - | off | - | - | - | - | in progress |
| LangMem | - | - | on | - | - | - | - | in progress |

---

## Setup

### 1. Conda Environment

```bash
conda create -n mirix python=3.10
conda activate mirix
pip install mem0 langgraph langmem openai jinja2 python-dotenv tqdm qdrant-client
```

### 2. Dataset

Download `locomo10.json` from [here](https://github.com/snap-research/locomo/tree/main/data) and place it at `dataset/locomo10.json`.

`dataset/locomo10_rag.json` is a pre-processed flat version used by RAG and LangMem — already included.

### 3. Local Servers (for local techniques)

You need two servers running via SSH tunnel or locally:

| Server | Port | Purpose |
|---|---|---|
| vLLM (Qwen3-32B) | 8001 | LLM inference |
| Embed (BGE-base-en-v1.5) | 8002 | Embeddings |

SSH tunnel example (IIT Pkd cluster):
```bash
ssh -p 49122 -L 8001:gpu02:8001 -L 8002:gpu03:8002 <user>@madhava.iitpkd.ac.in -N
```

### 4. Qdrant (for `mem0_local` and `rag`)

Download the Qdrant binary from https://qdrant.tech/documentation/quick-start/ and run it from the project root so storage persists at `storage/`:

```bash
# Windows
.\qdrant.exe

# Linux/Mac
./qdrant
```

Qdrant will be available at `http://localhost:6333`. Data persists to disk — safe to restart.

### 5. `.env` File

```bash
# Local vLLM
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_MODEL=Qwen/Qwen3-32B

# Local embed server
EMBED_BASE_URL=http://localhost:8002/v1
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# Cloud keys (not used in local runs)
# OPENAI_API_KEY=
# MEM0_API_KEY=
# ZEP_API_KEY=
# MEMOBASE_API_KEY=
```

### 6. LangMem Patch

LangMem has a bug with `typing_extensions`. Fix it:

```bash
# Find the file
python -c "import langmem; print(langmem.__file__)"
# Navigate to knowledge/extraction.py in that folder
# Add this line alongside the existing 'from typing_extensions import TypedDict':
#   import typing_extensions
```

---

## Running Experiments

All commands run from the project root (`locomo-benchmark/`).

### mem0_local (fully local, recommended)

```bash
# Step 1 — Add memories to Qdrant (run once, persists to disk)
python run_experiments.py --technique_type mem0_local --method add

# Step 2 — Search and answer questions
python run_experiments.py --technique_type mem0_local --method search --top_k 10 --output_folder results/
```

> Add phase stores memories in Qdrant at `storage/`. Safe to stop and resume search later — Qdrant persists.

### LangMem (fully local)

```bash
python run_experiments.py --technique_type langmem --output_folder results/
```

> Uses `InMemoryStore` — memories live in RAM only. Resume works between conversations but not within a conversation. If stopped mid-conversation, delete that conversation's key from `results/langmem_local_results.json` before restarting.

### RAG (fully local)

```bash
# Dense search, fixed chunking (default)
python run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 3 --output_folder results/

# Sparse (BM25 only)
python run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 3 --config qwen3_qwen3_sparse --output_folder results/

# Hybrid (BM25 + dense)
python run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 3 --config qwen3_qwen3_hybrid --output_folder results/

# Turn-level chunking
python run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 3 --config qwen3_qwen3_turn --output_folder results/

# Multi-turn window (3 turns)
python run_experiments.py --technique_type rag --chunk_size 500 --num_chunks 3 --config qwen3_qwen3_multiturn --output_folder results/

# Full context (no chunking)
python run_experiments.py --technique_type rag --chunk_size -1 --num_chunks 1 --output_folder results/
```

---

## Evaluation

### Step 1 — LLM Judge

```bash
python evals.py --input_file results/mem0_local_results.json --output_file results/mem0_local_evals.json
```

### Step 2 — Generate Scores

```bash
python generate_scores.py --input_path results/mem0_local_evals.json
```

Example output:
```
Mean Scores Per Category:
          bleu_score  f1_score  llm_score  count         type
category
1             0.3516    0.4629     0.7092    282   single_hop
2             0.4758    0.6423     0.8505    321     temporal
3             0.1758    0.2293     0.4688     96    multi_hop
4             0.4089    0.5155     0.7717    841  open_domain

Overall Mean Scores:
bleu_score    0.3978
f1_score      0.5145
llm_score     0.7578
```

---

## Project Structure

```
locomo-benchmark/
├── dataset/
│   ├── locomo10.json           # Original LoComo dataset (10 conversations)
│   └── locomo10_rag.json       # Pre-processed flat format for RAG/LangMem
├── src/
│   ├── langmem.py              # LangMem (local vLLM + InMemoryStore)
│   ├── rag.py                  # RAG (Qdrant, fixed/turn/multi-turn, dense/sparse/hybrid)
│   ├── memzero/
│   │   ├── add_local.py        # mem0 OSS add (local vLLM + Qdrant)
│   │   └── search_local.py     # mem0 OSS search (local vLLM + Qdrant)
├── metrics/
│   ├── llm_judge.py            # LLM judge scorer
│   └── utils.py
├── results/                    # Output JSON files from each technique
├── fixture/memobase/           # Pre-run Memobase artifacts
├── configs.py                  # RAG experiment configs (model, search type, chunking)
├── prompts.py                  # Answer + judge prompt templates
├── run_experiments.py          # Main entry point
├── evals.py                    # LLM judge evaluation
├── generate_scores.py          # Score aggregation
├── Makefile                    # Shortcut commands
└── .env                        # API keys and server URLs
```

---

## RAG Configs (`configs.py`)

| Config key | Search | Chunking | Thinking |
|---|---|---|---|
| `qwen3_qwen3_nothink` | dense | fixed | off |
| `qwen3_qwen3_think` | dense | fixed | on |
| `qwen3_qwen3_sparse` | sparse (BM25) | fixed | off |
| `qwen3_qwen3_hybrid` | hybrid | fixed | off |
| `qwen3_qwen3_turn` | dense | turn-level | off |
| `qwen3_qwen3_multiturn` | dense | multi-turn (3) | off |
| `llama7b_qwen3_nothink` | dense | fixed | off |

Pass with `--config <key>` to `run_experiments.py`.
