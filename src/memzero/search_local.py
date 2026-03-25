"""
Mem0 OSS local search — uses the open-source `Memory` class (no cloud API key).
Drop-in replacement for search.py that runs entirely on your local vLLM + embed server.
"""
import os
import json
import time
from collections import defaultdict
from tqdm import tqdm
from jinja2 import Template
from openai import OpenAI
from mem0 import Memory
from dotenv import load_dotenv
from prompts import ANSWER_PROMPT

load_dotenv()

VLLM_BASE_URL  = os.getenv("VLLM_BASE_URL",  "http://localhost:8001/v1")
VLLM_MODEL     = os.getenv("VLLM_MODEL",     "Qwen/Qwen3-32B")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:8002/v1")
EMBED_MODEL    = os.getenv("EMBEDDING_MODEL","BAAI/bge-base-en-v1.5")

MEM0_CONFIG = {
    "llm": {
        "provider": "openai",
        "config": {
            "model":    VLLM_MODEL,
            "openai_base_url": VLLM_BASE_URL,
            "api_key":  "EMPTY",
            "temperature": 0.0,
        },
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model":    EMBED_MODEL,
            "openai_base_url": EMBED_BASE_URL,
            "api_key":  "EMPTY",
            "embedding_dims": 768,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "locomo_mem0_local",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768,
        },
    },
}

ANSWER_PROMPT_TEMPLATE = Template(ANSWER_PROMPT)


class MemorySearchLocal:
    def __init__(self, output_path="results/mem0_local_results.json", top_k=10):
        self.memory       = Memory.from_config(MEM0_CONFIG)
        self.top_k        = top_k
        self.output_path  = output_path
        self.results      = defaultdict(list)
        # Answer LLM — local vLLM
        self.llm_client   = OpenAI(api_key="EMPTY", base_url=VLLM_BASE_URL)

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        t1 = time.time()
        for attempt in range(max_retries):
            try:
                memories = self.memory.search(query, user_id=user_id, limit=self.top_k)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise e
        t2 = time.time()

        # Normalise — OSS Memory.search returns list of dicts
        results_list = memories if isinstance(memories, list) else memories.get("results", [])
        semantic_memories = [
            {
                "memory":    m.get("memory", ""),
                "timestamp": m.get("metadata", {}).get("timestamp", ""),
                "score":     round(m.get("score", 0.0), 2),
            }
            for m in results_list
        ]
        return semantic_memories, t2 - t1

    def answer_question(self, speaker_a_uid, speaker_b_uid, question):
        mems_a, time_a = self.search_memory(speaker_a_uid, question)
        mems_b, time_b = self.search_memory(speaker_b_uid, question)

        fmt_a = [f"{m['timestamp']}: {m['memory']}" for m in mems_a]
        fmt_b = [f"{m['timestamp']}: {m['memory']}" for m in mems_b]

        answer_prompt = ANSWER_PROMPT_TEMPLATE.render(
            speaker_1_user_id=speaker_a_uid.rsplit("_", 1)[0],
            speaker_2_user_id=speaker_b_uid.rsplit("_", 1)[0],
            speaker_1_memories=json.dumps(fmt_a, indent=4),
            speaker_2_memories=json.dumps(fmt_b, indent=4),
            question=question,
        )

        t1 = time.time()
        response = self.llm_client.chat.completions.create(
            model=VLLM_MODEL,
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0,
        )
        t2 = time.time()

        return (
            response.choices[0].message.content,
            mems_a, mems_b,
            time_a, time_b,
            t2 - t1,
        )

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        # Resume: load existing results
        if os.path.exists(self.output_path):
            with open(self.output_path, "r") as f:
                existing = json.load(f)
            self.results.update({int(k): v for k, v in existing.items()})
            print(f"Resuming: loaded {len(self.results)} completed conversations")

        for idx, item in tqdm(enumerate(data), total=len(data),
                               desc="Processing conversations", colour="green"):
            qa           = item["qa"]
            conversation = item["conversation"]
            speaker_a    = conversation["speaker_a"]
            speaker_b    = conversation["speaker_b"]
            uid_a        = f"{speaker_a}_{idx}"
            uid_b        = f"{speaker_b}_{idx}"

            already_done = len(self.results.get(idx, []))
            non_cat5     = [q for q in qa if int(q.get("category", -1)) != 5]
            if already_done >= len(non_cat5):
                print(f"  Skipping conv {idx} (already complete)")
                continue
            if already_done > 0:
                print(f"  Resuming conv {idx} from question {already_done}/{len(non_cat5)}")

            for q_item in tqdm(qa[already_done:], desc=f"Questions [{idx}]", leave=False, colour="yellow"):
                category         = q_item.get("category", -1)
                if int(category) == 5:
                    continue

                question         = q_item.get("question", "")
                answer           = q_item.get("answer", "")
                evidence         = q_item.get("evidence", [])
                adversarial_ans  = q_item.get("adversarial_answer", "")

                (response, mems_a, mems_b,
                 time_a, time_b, resp_time) = self.answer_question(
                    uid_a, uid_b, question
                )

                self.results[idx].append({
                    "question":               question,
                    "answer":                 answer,
                    "category":               category,
                    "evidence":               evidence,
                    "adversarial_answer":     adversarial_ans,
                    "response":               response,
                    "speaker_1_memories":     mems_a,
                    "speaker_2_memories":     mems_b,
                    "num_speaker_1_memories": len(mems_a),
                    "num_speaker_2_memories": len(mems_b),
                    "speaker_1_memory_time":  time_a,
                    "speaker_2_memory_time":  time_b,
                    "response_time":          resp_time,
                })

                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)

        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)


if __name__ == "__main__":
    searcher = MemorySearchLocal(output_path="results/mem0_local_results.json")
    searcher.process_data_file("dataset/locomo10.json")
