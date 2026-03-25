"""
Experiment configurations for LoComo benchmark.

Naming convention: {rag_model}_{judge_model}_{thinking_mode}
"""

CONFIGS = {
    # Qwen3-32B as both RAG and judge, thinking disabled
    "qwen3_qwen3_nothink": {
        "rag_model": "Qwen/Qwen3-32B",
        "judge_model": "Qwen/Qwen3-32B",
        "rag_base_url": "http://localhost:8001/v1",
        "judge_base_url": "http://localhost:8001/v1",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "embed_base_url": "http://localhost:8002/v1",
        "enable_thinking": False,
        "max_tokens": 100,
        "judge_max_tokens": 20,
        "search_type": "dense",
        "alpha": 0.5,
        "chunk_strategy": "fixed",
        "output_prefix": "rag_qwen3_qwen3_nothink",
    },

    # Qwen3-32B think RAG, nothink judge (think judge produces malformed JSON)
    "qwen3_think_judged_nothink": {
        "rag_model": "Qwen/Qwen3-32B",
        "judge_model": "Qwen/Qwen3-32B",
        "rag_base_url": "http://localhost:8001/v1",
        "judge_base_url": "http://localhost:8001/v1",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "embed_base_url": "http://localhost:8002/v1",
        "enable_thinking": False,
        "max_tokens": 100,
        "judge_max_tokens": 20,
        "search_type": "dense",
        "alpha": 0.5,
        "chunk_strategy": "fixed",
        "output_prefix": "rag_qwen3_qwen3_think",
    },

    # Qwen3-32B as both RAG and judge, thinking enabled
    "qwen3_qwen3_think": {
        "rag_model": "Qwen/Qwen3-32B",
        "judge_model": "Qwen/Qwen3-32B",
        "rag_base_url": "http://localhost:8001/v1",
        "judge_base_url": "http://localhost:8001/v1",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "embed_base_url": "http://localhost:8002/v1",
        "enable_thinking": True,
        "max_tokens": 1500,
        "judge_max_tokens": 1500,
        "search_type": "dense",
        "alpha": 0.5,
        "chunk_strategy": "fixed",
        "output_prefix": "rag_qwen3_qwen3_think",
    },

    # Llama-7B as RAG, Qwen3-32B as judge, thinking disabled
    "llama7b_qwen3_nothink": {
        "rag_model": "meta-llama/Llama-3.1-8B-Instruct",
        "judge_model": "Qwen/Qwen3-32B",
        "rag_base_url": "http://localhost:8003/v1",
        "judge_base_url": "http://localhost:8001/v1",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "embed_base_url": "http://localhost:8002/v1",
        "enable_thinking": False,
        "max_tokens": 100,
        "judge_max_tokens": 20,
        "search_type": "dense",
        "alpha": 0.5,
        "chunk_strategy": "fixed",
        "output_prefix": "rag_llama7b_qwen3_nothink",
    },

    # Qwen3-32B, sparse (BM25 only)
    "qwen3_qwen3_sparse": {
        "rag_model": "Qwen/Qwen3-32B",
        "judge_model": "Qwen/Qwen3-32B",
        "rag_base_url": "http://localhost:8001/v1",
        "judge_base_url": "http://localhost:8001/v1",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "embed_base_url": "http://localhost:8002/v1",
        "enable_thinking": False,
        "max_tokens": 100,
        "judge_max_tokens": 20,
        "search_type": "sparse",
        "alpha": 0.0,
        "chunk_strategy": "fixed",
        "output_prefix": "rag_qwen3_qwen3_sparse",
    },

    # Qwen3-32B, hybrid (BM25 + dense)
    "qwen3_qwen3_hybrid": {
        "rag_model": "Qwen/Qwen3-32B",
        "judge_model": "Qwen/Qwen3-32B",
        "rag_base_url": "http://localhost:8001/v1",
        "judge_base_url": "http://localhost:8001/v1",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "embed_base_url": "http://localhost:8002/v1",
        "enable_thinking": False,
        "max_tokens": 100,
        "judge_max_tokens": 20,
        "search_type": "hybrid",
        "alpha": 0.5,
        "chunk_strategy": "fixed",
        "output_prefix": "rag_qwen3_qwen3_hybrid",
    },

    # Qwen3-32B, turn-level chunking, dense
    "qwen3_qwen3_turn": {
        "rag_model": "Qwen/Qwen3-32B",
        "judge_model": "Qwen/Qwen3-32B",
        "rag_base_url": "http://localhost:8001/v1",
        "judge_base_url": "http://localhost:8001/v1",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "embed_base_url": "http://localhost:8002/v1",
        "enable_thinking": False,
        "max_tokens": 100,
        "judge_max_tokens": 20,
        "search_type": "dense",
        "alpha": 0.5,
        "chunk_strategy": "turn",
        "output_prefix": "rag_qwen3_qwen3_turn",
    },

    # Qwen3-32B, multi-turn window (3 turns), dense
    "qwen3_qwen3_multiturn": {
        "rag_model": "Qwen/Qwen3-32B",
        "judge_model": "Qwen/Qwen3-32B",
        "rag_base_url": "http://localhost:8001/v1",
        "judge_base_url": "http://localhost:8001/v1",
        "embed_model": "BAAI/bge-base-en-v1.5",
        "embed_base_url": "http://localhost:8002/v1",
        "enable_thinking": False,
        "max_tokens": 100,
        "judge_max_tokens": 20,
        "search_type": "dense",
        "alpha": 0.5,
        "chunk_strategy": "multi_turn",
        "turn_window": 3,
        "output_prefix": "rag_qwen3_qwen3_multiturn",
    },
}

# Default config to use
DEFAULT_CONFIG = "qwen3_qwen3_nothink"
