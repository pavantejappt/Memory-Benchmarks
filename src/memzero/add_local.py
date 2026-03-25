"""
Mem0 OSS local add — uses the open-source `Memory` class (no cloud API key).
Drop-in replacement for add.py that runs entirely on your local vLLM + embed server.
"""
import os
import json
import time
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from mem0 import Memory

load_dotenv()

VLLM_BASE_URL  = os.getenv("VLLM_BASE_URL",  "http://localhost:8001/v1")
VLLM_MODEL     = os.getenv("VLLM_MODEL",     "Qwen/Qwen3-32B")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:8002/v1")
EMBED_MODEL    = os.getenv("EMBEDDING_MODEL","BAAI/bge-base-en-v1.5")

# Mem0 OSS config — points every call to local servers
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
    # In-memory vector store — no external DB needed
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

CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name (do not use "user")
   - Personal details (career, hobbies, life circumstances)
   - Emotional states and reactions
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family and relationships
   - Creative outlets and hobbies
   - Career aspirations and milestones

3. Make each memory rich with specific details rather than general statements.
   Include timeframes (exact dates when possible).

4. Extract memories only from user messages, not assistant responses.

5. Format each memory as a clear narrative paragraph.
"""


class MemoryADDLocal:
    def __init__(self, data_path=None, batch_size=2):
        self.memory = Memory.from_config(MEM0_CONFIG)
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, messages, metadata, retries=3):
        for attempt in range(retries):
            try:
                self.memory.add(
                    messages,
                    user_id=user_id,
                    metadata=metadata,
                )
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc, colour="cyan"):
            batch = messages[i: i + self.batch_size]
            self.add_memory(speaker, batch, metadata={"timestamp": timestamp})

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_uid = f"{speaker_a}_{idx}"
        speaker_b_uid = f"{speaker_b}_{idx}"

        # Delete existing memories for clean run
        try:
            self.memory.delete_all(user_id=speaker_a_uid)
            self.memory.delete_all(user_id=speaker_b_uid)
        except Exception:
            pass  # ok if nothing exists yet

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages_a, messages_b = [], []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages_a.append({"role": "user",      "content": f"{speaker_a}: {chat['text']}"})
                    messages_b.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages_a.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_b.append({"role": "user",      "content": f"{speaker_b}: {chat['text']}"})

            t_a = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_a_uid, messages_a, timestamp, f"Speaker A [{idx}]"),
            )
            t_b = threading.Thread(
                target=self.add_memories_for_speaker,
                args=(speaker_b_uid, messages_b, timestamp, f"Speaker B [{idx}]"),
            )
            t_a.start(); t_b.start()
            t_a.join();  t_b.join()

        print(f"Conversation {idx}: memories added.")

    def process_all_conversations(self, max_workers=4):
        if not self.data:
            raise ValueError("No data loaded.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.process_conversation, item, idx)
                for idx, item in enumerate(self.data)
            ]
            for f in futures:
                f.result()
