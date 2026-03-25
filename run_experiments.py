import os
import json
import argparse
from src.utils import TECHNIQUES, METHODS


class Experiment:
    def __init__(self, technique_type, chunk_size):
        self.technique_type = technique_type
        self.chunk_size = chunk_size

    def run(self):
        print(
            f"Running experiment with technique: {self.technique_type}, "
            f"chunk size: {self.chunk_size}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument(
        "--technique_type",
        choices=TECHNIQUES,
        default="rag",
        help="Memory technique to use",
    )
    parser.add_argument(
        "--method", choices=METHODS, default="add", help="Method to use (add|search)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=1000, help="Chunk size for RAG"
    )
    parser.add_argument(
        "--output_folder", type=str, default="results/", help="Output folder"
    )
    parser.add_argument(
        "--top_k", type=int, default=30, help="Top-k memories to retrieve"
    )
    parser.add_argument(
        "--filter_memories", action="store_true", default=False,
    )
    parser.add_argument(
        "--is_graph", action="store_true", default=False,
    )
    parser.add_argument(
        "--num_chunks", type=int, default=1, help="Number of chunks (RAG k)"
    )
    parser.add_argument(
        "--config", type=str, default="qwen3_qwen3_nothink",
        help="Config name from configs.py",
    )

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    print(
        f"Running: technique={args.technique_type}  "
        f"method={args.method}  chunk_size={args.chunk_size}"
    )

    # ── RAG ──────────────────────────────────────────────────────────────────
    if args.technique_type == "rag":
        from src.rag import RAGManager
        from configs import CONFIGS

        cfg = CONFIGS[args.config]
        output_file_path = os.path.join(
            args.output_folder,
            f"{cfg['output_prefix']}_{args.chunk_size}_k{args.num_chunks}.json",
        )
        rag_manager = RAGManager(
            data_path="dataset/locomo10_rag.json",
            chunk_size=args.chunk_size,
            k=args.num_chunks,
            config_name=args.config,
        )
        rag_manager.process_all_conversations(output_file_path)

    # ── LangMem (fully local) ─────────────────────────────────────────────────
    elif args.technique_type == "langmem":
        from src.langmem import LangMemManager

        output_file_path = os.path.join(args.output_folder, "langmem_local_results.json")
        mgr = LangMemManager(dataset_path="dataset/locomo10_rag.json")
        mgr.process_all_conversations(output_file_path)

    # ── Mem0 OSS (fully local) ────────────────────────────────────────────────
    elif args.technique_type == "mem0_local":
        from src.memzero.add_local import MemoryADDLocal
        from src.memzero.search_local import MemorySearchLocal

        if args.method == "add":
            mgr = MemoryADDLocal(data_path="dataset/locomo10.json")
            mgr.process_all_conversations()
        elif args.method == "search":
            output_file_path = os.path.join(
                args.output_folder,
                f"mem0_local_results_top_{args.top_k}.json",
            )
            mgr = MemorySearchLocal(output_path=output_file_path, top_k=args.top_k)
            mgr.process_data_file("dataset/locomo10.json")

    # ── Mem0 cloud (original — needs MEM0_API_KEY) ────────────────────────────
    elif args.technique_type == "mem0":
        from src.memzero.add import MemoryADD
        from src.memzero.search import MemorySearch

        if args.method == "add":
            mgr = MemoryADD(data_path="dataset/locomo10.json", is_graph=args.is_graph)
            mgr.process_all_conversations()
        elif args.method == "search":
            output_file_path = os.path.join(
                args.output_folder,
                f"mem0_results_top_{args.top_k}_filter_{args.filter_memories}"
                f"_graph_{args.is_graph}.json",
            )
            mgr = MemorySearch(output_file_path, args.top_k,
                               args.filter_memories, args.is_graph)
            mgr.process_data_file("dataset/locomo10.json")

    # ── Zep cloud (original — needs ZEP_API_KEY) ──────────────────────────────
    elif args.technique_type == "zep":
        from src.zep.search import ZepSearch
        from src.zep.add import ZepAdd

        if args.method == "add":
            mgr = ZepAdd(data_path="dataset/locomo10.json")
            mgr.process_all_conversations("1")
        elif args.method == "search":
            output_file_path = os.path.join(args.output_folder, "zep_search_results.json")
            mgr = ZepSearch()
            mgr.process_data_file("dataset/locomo10.json", "1", output_file_path)

    # ── OpenAI (original — needs OPENAI_API_KEY) ──────────────────────────────
    elif args.technique_type == "openai":
        from src.openai.predict import OpenAIPredict

        output_file_path = os.path.join(args.output_folder, "openai_results.json")
        mgr = OpenAIPredict()
        mgr.process_data_file("dataset/locomo10.json", output_file_path)

    # ── Memobase cloud (original) ─────────────────────────────────────────────
    elif args.technique_type == "memobase":
        from src.memobase_client import MemobaseADD, MemobaseSearch

        if args.method == "add":
            mgr = MemobaseADD(data_path="dataset/locomo10.json")
            mgr.process_all_conversations()
        elif args.method == "search":
            mgr = MemobaseSearch()
            mgr.process_data_file("dataset/locomo10.json")

    else:
        raise ValueError(f"Invalid technique type: {args.technique_type}")


if __name__ == "__main__":
    main()
