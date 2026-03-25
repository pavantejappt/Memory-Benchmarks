"""
Runs RAG benchmark + LLM judge for all specified configs sequentially.
"""
import os
import subprocess
import sys
from configs import CONFIGS

CHUNK_SIZE = 500
NUM_CHUNKS = 3
OUTPUT_FOLDER = "results/"

# Configs to run
RUN_CONFIGS = [
    "qwen3_qwen3_nothink",
    "qwen3_qwen3_think",
    "qwen3_qwen3_sparse",
    "qwen3_qwen3_hybrid",
]


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


if __name__ == "__main__":
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for config_name in RUN_CONFIGS:
        cfg = CONFIGS[config_name]
        results_file = os.path.join(
            OUTPUT_FOLDER,
            f"{cfg['output_prefix']}_{CHUNK_SIZE}_k{NUM_CHUNKS}.json"
        )

        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name}")
        print(f"RAG model: {cfg['rag_model']} | thinking: {cfg['enable_thinking']}")
        print(f"Output: {results_file}")
        print(f"{'='*60}")

        # Step 1: Run RAG benchmark
        run([
            sys.executable, "run_experiments.py",
            "--technique_type", "rag",
            "--chunk_size", str(CHUNK_SIZE),
            "--num_chunks", str(NUM_CHUNKS),
            "--config", config_name,
            "--output_folder", OUTPUT_FOLDER,
        ])

        # Step 2: Run LLM judge on results
        run([
            sys.executable, "metrics/llm_judge.py",
            "--input_file", results_file,
            "--config", config_name,
        ])

        print(f"\n✓ Done: {config_name}")

    print(f"\n{'='*60}")
    print("All configs completed.")
