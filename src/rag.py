from openai import OpenAI
import json
import numpy as np
from tqdm import tqdm
from jinja2 import Template
import time
from collections import defaultdict
import os
import re
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from configs import CONFIGS, DEFAULT_CONFIG

load_dotenv()

PROMPT = """
# Question: 
{{QUESTION}}

# Context: 
{{CONTEXT}}

# Short answer:
"""


class RAGManager:
    def __init__(self, data_path="dataset/locomo10_rag.json", chunk_size=500, k=1, config_name=DEFAULT_CONFIG):
        cfg = CONFIGS[config_name]
        self.model = cfg["rag_model"]
        self.enable_thinking = cfg["enable_thinking"]
        self.max_tokens = cfg["max_tokens"]
        self.search_type = cfg.get("search_type", "dense")  # dense, sparse, hybrid
        self.alpha = cfg.get("alpha", 0.5)  # weight for dense in hybrid
        self.chunk_strategy = cfg.get("chunk_strategy", "fixed")  # fixed, turn, multi_turn
        self.turn_window = cfg.get("turn_window", 3)  # window size for multi_turn
        self.client = OpenAI(api_key="EMPTY", base_url=cfg["rag_base_url"])
        self.embed_client = OpenAI(api_key="EMPTY", base_url=cfg["embed_base_url"])
        self.embed_model = cfg["embed_model"]
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k

    def generate_response(self, question, context):
        template = Template(PROMPT)
        prompt = template.render(
            CONTEXT=context,
            QUESTION=question
        )

        max_retries = 3
        retries = 0

        while retries <= max_retries:
            try:
                t1 = time.time()
                kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "You are a helpful assistant that can answer "
                                    "questions based on the provided context."
                                    "If the question involves timing, use the conversation date for reference."
                                    "Provide the shortest possible answer."
                                    "Use words directly from the conversation when possible."
                                    "Avoid using subjects in your answer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
                )
                if self.max_tokens is not None:
                    kwargs["max_tokens"] = self.max_tokens
                response = self.client.chat.completions.create(**kwargs)
                t2 = time.time()
                content = response.choices[0].message.content
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content, t2-t1
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    raise e
                time.sleep(1)  # Wait before retrying

    def clean_chat_history(self, chat_history):
        cleaned_chat_history = ""
        for c in chat_history:
            cleaned_chat_history += (f"{c['timestamp']} | {c['speaker']}: "
                                     f"{c['text']}\n")

        return cleaned_chat_history

    def calculate_embedding(self, document):
        response = self.embed_client.embeddings.create(
            model=self.embed_model,
            input=document
        )
        return response.data[0].embedding

    def calculate_similarity(self, embedding1, embedding2):
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    def search(self, query, chunks, embeddings, bm25, k=1):
        t1 = time.time()

        if self.search_type == "sparse":
            tokenized_query = query.lower().split()
            scores = bm25.get_scores(tokenized_query)
            top_indices = np.argsort(scores)[-k:][::-1]

        elif self.search_type == "hybrid":
            # Dense scores (normalized)
            query_embedding = self.calculate_embedding(query)
            dense_scores = np.array([self.calculate_similarity(query_embedding, e) for e in embeddings])
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-9)

            # Sparse scores (normalized)
            tokenized_query = query.lower().split()
            sparse_scores = np.array(bm25.get_scores(tokenized_query))
            sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-9)

            combined = self.alpha * dense_scores + (1 - self.alpha) * sparse_scores
            top_indices = np.argsort(combined)[-k:][::-1]

        else:  # dense (default)
            query_embedding = self.calculate_embedding(query)
            dense_scores = np.array([self.calculate_similarity(query_embedding, e) for e in embeddings])
            top_indices = np.argsort(dense_scores)[-k:][::-1]

        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])
        t2 = time.time()
        return combined_chunks, t2 - t1

    def create_chunks(self, chat_history, chunk_size=500):
        if chunk_size == -1:
            return [self.clean_chat_history(chat_history)], [], None

        if self.chunk_strategy == "turn":
            chunks = [
                f"{c['timestamp']} | {c['speaker']}: {c['text']}"
                for c in chat_history
            ]
        elif self.chunk_strategy == "multi_turn":
            w = self.turn_window
            chunks = [
                "\n".join(f"{c['timestamp']} | {c['speaker']}: {c['text']}" for c in chat_history[i:i+w])
                for i in range(0, len(chat_history), w - 1)
            ]
        else:  # fixed
            documents = self.clean_chat_history(chat_history)
            char_size = chunk_size * 4
            chunks = [documents[i:i+char_size] for i in range(0, len(documents), char_size)]

        # Build BM25 index
        tokenized_chunks = [c.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_chunks)

        # Batch all chunks in a single HTTP call (skip if sparse only)
        if self.search_type != "sparse":
            response = self.embed_client.embeddings.create(model=self.embed_model, input=chunks)
            embeddings = [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
        else:
            embeddings = []

        print(f"  → {len(chunks)} chunks | search: {self.search_type}")
        return chunks, embeddings, bm25

    def process_all_conversations(self, output_file_path):
        with open(self.data_path, "r") as f:
            data = json.load(f)

        FINAL_RESULTS = defaultdict(list)
        # Resume: load existing results if file exists
        if os.path.exists(output_file_path):
            with open(output_file_path, "r") as f:
                existing = json.load(f)
            FINAL_RESULTS.update(existing)
            print(f"Resuming: loaded {len(FINAL_RESULTS)} completed conversations")

        for key, value in tqdm(data.items(), desc="Processing conversations"):
            chat_history = value["conversation"]
            questions = value["question"]

            # Skip if already fully answered
            total_questions = len(value["question"])
            already_done = len(FINAL_RESULTS.get(key, []))
            if already_done >= total_questions:
                print(f"  Skipping conversation {key} (already complete)")
                continue

            chunks, embeddings, bm25 = self.create_chunks(
                chat_history, self.chunk_size
            )

            if already_done > 0:
                print(f"  Resuming conversation {key} from question {already_done}/{total_questions}")

            for item in tqdm(
                questions[already_done:], desc="Answering questions", leave=False
            ):
                question = item["question"]
                answer = item.get("answer", "")
                category = item["category"]

                if int(category) == 5:
                    continue

                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                else:
                    context, search_time = self.search(
                        question, chunks, embeddings, bm25, k=self.k
                    )
                response, response_time = self.generate_response(
                    question, context
                )

                FINAL_RESULTS[key].append({
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "context": context,
                    "response": response,
                    "search_time": search_time,
                    "response_time": response_time,
                })
                with open(output_file_path, "w+") as f:
                    json.dump(FINAL_RESULTS, f, indent=4)

        # Save results
        with open(output_file_path, "w+") as f:
            json.dump(FINAL_RESULTS, f, indent=4)
