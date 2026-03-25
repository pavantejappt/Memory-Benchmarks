# ─────────────────────────────────────────────────────────────────────────────
# Set OpenAI-compat env vars BEFORE any langchain/langmem imports so every
# internal OpenAI client they create picks up the local vLLM endpoint.
# ─────────────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

load_dotenv()

# Override to local vLLM — these must come before langgraph/langmem imports
os.environ["OPENAI_API_KEY"] = "EMPTY"
os.environ["OPENAI_BASE_URL"] = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:8002/v1")
EMBED_MODEL    = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
LLM_MODEL      = os.getenv("VLLM_MODEL", "Qwen/Qwen3-32B")

# ─────────────────────────────────────────────────────────────────────────────
# Now safe to import langgraph / langmem
# ─────────────────────────────────────────────────────────────────────────────
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.utils.config import get_store
from langmem import (
    create_manage_memory_tool,
    create_search_memory_tool,
)

import time
import json
import os
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI
from collections import defaultdict
from jinja2 import Template
from prompts import ANSWER_PROMPT

# ─────────────────────────────────────────────────────────────────────────────
# Answer client — points to local vLLM
# ─────────────────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key="EMPTY",
    base_url=os.environ["OPENAI_BASE_URL"],
)

ANSWER_PROMPT_TEMPLATE = Template(ANSWER_PROMPT)


def get_answer(question, speaker_1_user_id, speaker_1_memories,
               speaker_2_user_id, speaker_2_memories):
    prompt_text = ANSWER_PROMPT_TEMPLATE.render(
        question=question,
        speaker_1_user_id=speaker_1_user_id,
        speaker_1_memories=speaker_1_memories,
        speaker_2_user_id=speaker_2_user_id,
        speaker_2_memories=speaker_2_memories,
    )
    t1 = time.time()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": prompt_text}],
        temperature=0.0,
        max_tokens=250,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    t2 = time.time()
    return response.choices[0].message.content, t2 - t1


def prompt(state):
    """Prepare messages for the agent LLM."""
    store = get_store()
    memories = store.search(
        ("memories",),
        query=state["messages"][-1].content,
    )
    system_msg = (
        "You are a helpful assistant.\n\n"
        "## Memories\n<memories>\n"
        f"{memories}\n</memories>\n"
    )
    return [{"role": "system", "content": system_msg}, *state["messages"]]


# ─────────────────────────────────────────────────────────────────────────────
# Custom embed function — calls the local BGE embed server
# ─────────────────────────────────────────────────────────────────────────────
def _local_embed(texts: list) -> list:
    """Call local embed server (OpenAI-compat) and return list of vectors."""
    embed_client = OpenAI(api_key="EMPTY", base_url=EMBED_BASE_URL)
    response = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]


class LangMem:
    def __init__(self):
        # BGE-base-en-v1.5 → 768 dims (NOT 1536 like OpenAI ada-002)
        self.store = InMemoryStore(
            index={
                "dims": 768,
                "embed": _local_embed,
            }
        )
        self.checkpointer = MemorySaver()

        # create_react_agent reads OPENAI_BASE_URL + OPENAI_API_KEY from env
        self.agent = create_react_agent(
            f"openai:{LLM_MODEL}",
            prompt=prompt,
            tools=[
                create_manage_memory_tool(namespace=("memories",)),
                create_search_memory_tool(namespace=("memories",)),
            ],
            store=self.store,
            checkpointer=self.checkpointer,
        )

    def add_memory(self, message):
        # Use a unique thread_id each time so chat history never accumulates
        config = {"configurable": {"thread_id": str(time.time_ns())}}
        return self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )

    def search_memory(self, query, config):
        t1 = time.time()
        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                config=config,
            )
            t2 = time.time()
            return response["messages"][-1].content, t2 - t1
        except Exception as e:
            t2 = time.time()
            print(f"Error in search_memory: {e}")
            return "", t2 - t1


class LangMemManager:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        with open(self.dataset_path, "r") as f:
            self.data = json.load(f)

    def process_all_conversations(self, output_file_path):
        OUTPUT = defaultdict(list)

        # Resume: load existing results
        if os.path.exists(output_file_path):
            with open(output_file_path, "r") as f:
                OUTPUT.update(json.load(f))
            print(f"Resuming: loaded {len(OUTPUT)} completed conversations")

        for key, value in tqdm(self.data.items(), desc="Processing conversations", colour="green"):
            if key in OUTPUT and len(OUTPUT[key]) > 0:
                print(f"  Skipping conversation {key} (already complete)")
                continue

            chat_history = value["conversation"]
            questions    = value["question"]

            agent1 = LangMem()
            agent2 = LangMem()
            config = {"configurable": {"thread_id": f"thread-{key}"}}

            speakers_ordered = []
            for conv in chat_history:
                if conv["speaker"] not in speakers_ordered:
                    speakers_ordered.append(conv["speaker"])
                if len(speakers_ordered) == 2:
                    break

            if len(speakers_ordered) != 2:
                raise ValueError(f"Expected 2 speakers, got {len(speakers_ordered)}")

            speaker1, speaker2 = speakers_ordered[0], speakers_ordered[1]

            for conv in tqdm(chat_history, desc=f"Adding memories [{key}]", leave=False, colour="cyan"):
                message = f"{conv['timestamp']} | {conv['speaker']}: {conv['text']}"
                if conv["speaker"] == speaker1:
                    agent1.add_memory(message)
                else:
                    agent2.add_memory(message)

            for q in tqdm(questions, desc=f"Answering questions [{key}]", leave=False, colour="yellow"):
                category = q["category"]
                if int(category) == 5:
                    continue

                answer   = q["answer"]
                question = q["question"]

                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
                    f1 = ex.submit(agent1.search_memory, question, config)
                    f2 = ex.submit(agent2.search_memory, question, config)
                    response1, speaker1_memory_time = f1.result()
                    response2, speaker2_memory_time = f2.result()
                generated_answer, response_time = get_answer(
                    question, speaker1, response1, speaker2, response2
                )

                OUTPUT[key].append({
                    "question":             question,
                    "answer":               answer,
                    "response1":            response1,
                    "response2":            response2,
                    "category":             category,
                    "speaker1_memory_time": speaker1_memory_time,
                    "speaker2_memory_time": speaker2_memory_time,
                    "response_time":        response_time,
                    "response":             generated_answer,
                })

                # Save after every question
                with open(output_file_path, "w") as f:
                    json.dump(OUTPUT, f, indent=4)

        with open(output_file_path, "w") as f:
            json.dump(OUTPUT, f, indent=4)
