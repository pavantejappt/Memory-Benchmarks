import os
import sys
from openai import OpenAI
import json
import re
from collections import defaultdict
import numpy as np
import argparse
from dotenv import load_dotenv

# Add parent directory to path so configs can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import CONFIGS, DEFAULT_CONFIG

load_dotenv()


def get_client(config_name=DEFAULT_CONFIG):
    cfg = CONFIGS[config_name]
    return OpenAI(api_key="EMPTY", base_url=cfg["judge_base_url"]), cfg["judge_model"], cfg["enable_thinking"], cfg["judge_max_tokens"]

ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it’s time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


def evaluate_llm_judge(question, gold_answer, generated_answer, client, model, enable_thinking, max_tokens):
    """Evaluate the generated answer against the gold answer using an LLM judge."""
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": ACCURACY_PROMPT.format(
            question=question,
            gold_answer=gold_answer,
            generated_answer=generated_answer,
        )}],
        response_format={"type": "json_object"},
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    try:
        label = json.loads(content)["label"]
    except (KeyError, json.JSONDecodeError):
        # Fallback: scan for CORRECT/WRONG in raw content
        if "CORRECT" in content:
            label = "CORRECT"
        else:
            label = "WRONG"
    return 1 if label == "CORRECT" else 0


def main():
    """Main function to evaluate RAG results using LLM judge."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using LLM judge")
    parser.add_argument(
        "--input_file",
        type=str,
        default="results/default_run_v4_k30_new_graph.json",
        help="Path to the input dataset file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        choices=list(CONFIGS.keys()),
        help="Config name from configs.py",
    )

    args = parser.parse_args()
    client, model, enable_thinking, judge_max_tokens = get_client(args.config)

    dataset_path = args.input_file
    output_path = f"results/llm_judge_{dataset_path.split('/')[-1]}"

    with open(dataset_path, "r") as f:
        data = json.load(f)

    LLM_JUDGE = defaultdict(list)
    RESULTS = defaultdict(list)

    index = 0
    for k, v in data.items():
        for x in v:
            question = x["question"]
            gold_answer = x["answer"]
            generated_answer = x["response"]
            category = x["category"]

            # Skip category 5
            if int(category) == 5:
                continue

            # Evaluate the answer
            label = evaluate_llm_judge(question, gold_answer, generated_answer, client, model, enable_thinking, judge_max_tokens)
            LLM_JUDGE[category].append(label)

            # Store the results
            RESULTS[index].append(
                {
                    "question": question,
                    "gt_answer": gold_answer,
                    "response": generated_answer,
                    "category": category,
                    "llm_label": label,
                }
            )

            # Save intermediate results
            with open(output_path, "w") as f:
                json.dump(RESULTS, f, indent=4)

            # Print current accuracy for all categories
            print("All categories accuracy:")
            for cat, results in LLM_JUDGE.items():
                if results:  # Only print if there are results for this category
                    print(
                        f"  Category {cat}: {np.mean(results):.4f} "
                        f"({sum(results)}/{len(results)})"
                    )
            print("------------------------------------------")
        index += 1

    # Save final results
    with open(output_path, "w") as f:
        json.dump(RESULTS, f, indent=4)

    # Print final summary
    print("PATH: ", dataset_path)
    print("------------------------------------------")
    for k, v in LLM_JUDGE.items():
        print(k, np.mean(v))


if __name__ == "__main__":
    main()
