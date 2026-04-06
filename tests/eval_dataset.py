"""
Creates the evaluation dataset in LangSmith.
Run once: uv run python tests/eval_dataset.py

What it does:
- Creates a named dataset in your LangSmith project
- Adds question/answer pairs as examples
- Dataset is reused every time you run evals
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client

client = Client()
import urllib3
urllib3.disable_warnings()

import httpx
import langsmith

# Patch langsmith client to skip SSL verification
client = Client(
    api_key=os.environ.get("LANGSMITH_API_KEY"),
    api_url="https://api.smith.langchain.com",
)

# ── Dataset name ───────────────────────────────────────────────────────────────
DATASET_NAME = "langgraph-agent-qa"

# ── Question / answer pairs ────────────────────────────────────────────────────
# These are the questions your agent will be tested against.
# expected_answer is what a correct response should contain.
# Add your own based on documents you have ingested.

EXAMPLES = [
    # ── RAG questions (from company_policy.txt) ────────────────────────────────
    {
        "question": "What is the refund window for purchases?",
        "expected_answer": "30 days",
        "category": "rag",
    },
    {
        "question": "How long do refunds take to process?",
        "expected_answer": "5-7 business days",
        "category": "rag",
    },
    {
        "question": "Are digital products refundable?",
        "expected_answer": "No, digital products are non-refundable once downloaded",
        "category": "rag",
    },
    {
        "question": "How many days of annual leave do employees get?",
        "expected_answer": "20 days",
        "category": "rag",
    },
    {
        "question": "What is the meal allowance during business travel?",
        "expected_answer": "$50 per day",
        "category": "rag",
    },
    {
        "question": "How often are performance reviews conducted?",
        "expected_answer": "Twice a year, in June and December",
        "category": "rag",
    },

    # ── Calculator questions ───────────────────────────────────────────────────
    {
        "question": "What is 25 multiplied by 48?",
        "expected_answer": "1200",
        "category": "code",
    },
    {
        "question": "What is 15 percent of 200?",
        "expected_answer": "30",
        "category": "code",
    },

    # ── General knowledge questions ────────────────────────────────────────────
    {
        "question": "What is machine learning?",
        "expected_answer": "Machine learning is a branch of artificial intelligence that enables computers to learn from data",
        "category": "general",
    },
    {
        "question": "What is an API?",
        "expected_answer": "An API (Application Programming Interface) allows software applications to communicate with each other",
        "category": "general",
    },
]


def create_dataset():
    # ── Check if dataset already exists ───────────────────────────────────────
    existing = [d for d in client.list_datasets() if d.name == DATASET_NAME]

    if existing:
        print(f"Dataset '{DATASET_NAME}' already exists.")
        print(f"ID: {existing[0].id}")
        print(f"\nTo recreate it, delete it first in LangSmith UI")
        print(f"then run this script again.")

        # Show existing examples
        examples = list(client.list_examples(dataset_name=DATASET_NAME))
        print(f"\nExisting examples: {len(examples)}")
        for ex in examples:
            q = ex.inputs.get("question","")[:60]
            print(f"  - {q}...")
        return existing[0]

    # ── Create fresh dataset ───────────────────────────────────────────────────
    print(f"Creating dataset: '{DATASET_NAME}'")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="LangGraph agent quality evaluation — RAG, code, and general questions",
    )
    print(f"Created. ID: {dataset.id}")

    # ── Add examples ───────────────────────────────────────────────────────────
    print(f"\nAdding {len(EXAMPLES)} examples...")
    for i, ex in enumerate(EXAMPLES):
        client.create_example(
            dataset_id=dataset.id,
            inputs={"question": ex["question"]},
            outputs={"answer": ex["expected_answer"]},
            metadata={"category": ex["category"]},
        )
        print(f"  [{i+1}/{len(EXAMPLES)}] {ex['question'][:50]}...")

    print(f"\nDataset ready.")
    print(f"View at: https://smith.langchain.com/datasets/{dataset.id}")
    return dataset


if __name__ == "__main__":
    create_dataset()