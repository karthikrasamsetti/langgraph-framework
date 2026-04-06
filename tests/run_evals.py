"""
Runs evaluation against the LangSmith dataset.

Usage:
  # Default — uses settings from .env
  uv run python tests/run_evals.py

  # Custom experiment name
  uv run python tests/run_evals.py --name "bedrock-claude-v2-test"

  # Test specific category only
  uv run python tests/run_evals.py --category rag

What it does:
- Pulls questions from the LangSmith dataset
- Runs each question through your agent
- Scores each answer using LLM-as-judge
- Records results as an experiment in LangSmith
- Prints a summary with pass/fail per question
"""
import os
import sys

# ── Fix corporate network SSL certificate interception ────────────────────────
# Must be set before any other imports that make HTTPS calls
import certifi
os.environ["SSL_CERT_FILE"]      = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"]     = certifi.where()

import argparse
import requests
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

from config.settings import get_settings, parse_api_keys

settings     = get_settings()
client       = Client()
DATASET_NAME = "langgraph-agent-qa"


# ── Get API key for auth ───────────────────────────────────────────────────────
def get_eval_headers() -> dict:
    """Get auth headers for eval requests."""
    base = {"Content-Type": "application/json"}
    if not settings.AUTH_ENABLED:
        return base
    keys = parse_api_keys()
    if not keys:
        print("WARNING: AUTH_ENABLED but no API_KEYS found.")
        return base
    first_key = list(keys.keys())[0]
    return {**base, "X-API-Key": first_key}


# ── Agent runner ───────────────────────────────────────────────────────────────
def run_agent(inputs: dict) -> dict:
    """
    Runs one question through the agent.
    LangSmith calls this for each dataset example.
    """
    question = inputs["question"]
    api_base = f"http://localhost:{settings.API_PORT}"
    headers  = get_eval_headers()

    try:
        resp = requests.post(
            f"{api_base}/chat",
            json={
                "message":    question,
                "session_id": f"eval-{int(time.time())}",
            },
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        data          = resp.json()
        response_text = data.get("response") or data.get("final_response", "")
        error         = data.get("error", "")

        # ── Auto-approve human-in-loop pauses during evals ────────────────────
        if error and error.startswith("pending:"):
            session_id = data.get("session_id", "")
            print(f"  [auto-approve] session {session_id} paused at {error}")
            resume_resp = requests.post(
                f"{api_base}/chat/resume/{session_id}",
                json={"action": "approve", "mode": "single"},
                headers=headers,
                timeout=60,
            )
            if resume_resp.ok:
                resume_data   = resume_resp.json()
                response_text = (
                    resume_data.get("response")
                    or resume_data.get("final_response")
                    or ""
                )

        return {"answer": response_text or "No response returned."}

    except requests.exceptions.ConnectionError:
        return {"answer": "ERROR: Cannot connect to backend. Is uvicorn running?"}
    except Exception as e:
        return {"answer": f"ERROR: {e}"}


# ── LLM-as-judge evaluator ────────────────────────────────────────────────────
def correctness_evaluator(run: Run, example: Example) -> dict:
    """
    Uses an LLM to score whether the agent answer is correct.
    Score 0.0 to 1.0.
    """
    # ── SSL fix inside evaluator ───────────────────────────────────────────────
    import certifi
    os.environ["SSL_CERT_FILE"]      = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

    from langchain_core.messages import HumanMessage
    from llm.factory import get_llm

    question        = example.inputs.get("question", "")
    expected_answer = example.outputs.get("answer", "")
    actual_answer   = (run.outputs or {}).get("answer", "")

    if not actual_answer or not isinstance(actual_answer, str) or actual_answer.startswith("ERROR:"):
        return {"key": "correctness", "score": 0.0, "comment": actual_answer or "empty"}

    prompt = f"""You are evaluating an AI assistant's answer quality.

Question: {question}

Expected answer (key facts that should be present):
{expected_answer}

Actual answer given:
{actual_answer}

Score the actual answer from 0.0 to 1.0:
- 1.0: Correct and contains all key facts from expected answer
- 0.7: Mostly correct, contains most key facts
- 0.5: Partially correct, missing some key facts
- 0.3: Mostly incorrect
- 0.0: Wrong, refuses to answer, or is an error

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    try:
        llm      = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])
        score    = float(response.content.strip())
        score    = max(0.0, min(1.0, score))
        comment  = "PASS" if score >= 0.7 else "FAIL"
        return {"key": "correctness", "score": score, "comment": comment}
    except Exception as e:
        return {"key": "correctness", "score": 0.0, "comment": f"Evaluator error: {str(e)[:100]}"}

# ── Score extractor ────────────────────────────────────────────────────────────
def extract_score(result) -> tuple[float, str]:
    """
    Extracts score from result dict.
    Format confirmed: {'run': ..., 'example': ..., 'evaluation_results': {'results': [EvaluationResult(...)]}}
    """
    # Confirmed format from debug — dict with evaluation_results key containing dict with results list
    if isinstance(result, dict):
        er = result.get("evaluation_results", {})
        if isinstance(er, dict):
            results_list = er.get("results", [])
            if results_list:
                r = results_list[0]
                score   = r.score if r.score is not None else 0.0
                comment = r.comment or ""
                return float(score), comment

    return 0.0, "could not extract score"


def extract_question(result) -> str:
    """Extracts question using confirmed dict format."""
    if isinstance(result, dict):
        ex = result.get("example")
        if ex is not None:
            # Example is an object with inputs attribute
            if hasattr(ex, "inputs"):
                return ex.inputs.get("question", "")[:55]
            # Or a dict
            if isinstance(ex, dict):
                return ex.get("inputs", {}).get("question", "")[:55]
    return "unknown question"


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run LangSmith evals")
    parser.add_argument("--name",     type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--category", type=str, default=None,
                        choices=["rag", "code", "general"],
                        help="Only test this category")
    args = parser.parse_args()

    # ── Verify dataset exists ──────────────────────────────────────────────────
    datasets = [d for d in client.list_datasets() if d.name == DATASET_NAME]
    if not datasets:
        print(f"Dataset '{DATASET_NAME}' not found.")
        print(f"Run first: uv run python tests/eval_dataset.py")
        sys.exit(1)

    # ── Build experiment name ──────────────────────────────────────────────────
    if args.name:
        experiment_name = args.name
    elif settings.LLM_PROVIDER == "bedrock":
        model = settings.BEDROCK_MODEL_ID.split(".")[-1]
        experiment_name = f"bedrock-{model}-eval"
    elif settings.LLM_PROVIDER == "openai":
        experiment_name = f"openai-{settings.OPENAI_MODEL}-eval"
    elif settings.LLM_PROVIDER == "groq":
        experiment_name = f"groq-{settings.GROQ_MODEL}-eval"
    else:
        experiment_name = f"{settings.LLM_PROVIDER}-eval"

    if args.category:
        experiment_name += f"-{args.category}"

    print(f"\nRunning evals")
    print(f"  Dataset    : {DATASET_NAME}")
    print(f"  Experiment : {experiment_name}")
    print(f"  Provider   : {settings.LLM_PROVIDER}")
    print(f"  Category   : {args.category or 'all'}")
    print(f"  Auth       : {'enabled' if settings.AUTH_ENABLED else 'disabled'}")
    print()

    # ── Filter dataset by category ─────────────────────────────────────────────
    if args.category:
        examples = [
            ex for ex in client.list_examples(dataset_name=DATASET_NAME)
            if ex.metadata and ex.metadata.get("category") == args.category
        ]
        if not examples:
            print(f"No examples found for category '{args.category}'")
            sys.exit(1)
        print(f"  Filtered to {len(examples)} examples\n")
        data_source = examples
    else:
        data_source = DATASET_NAME

    # ── Run evaluation ─────────────────────────────────────────────────────────
    results = evaluate(
        run_agent,
        data=data_source,
        evaluators=[correctness_evaluator],
        experiment_prefix=experiment_name,
        metadata={
            "provider": settings.LLM_PROVIDER,
            "category": args.category or "all",
        },
        max_concurrency=1,
    )
    # ── Collect all results ────────────────────────────────────────────────────
    # evaluate() returns a lazy iterator — consume it fully first
    all_results = list(results)

    # # ── Debug — print first result structure ──────────────────────────────────
    # if all_results:
    #     first = all_results[0]
    #     print(f"\nDEBUG result type: {type(first)}")
    #     if isinstance(first, dict):
    #         print(f"DEBUG dict keys: {list(first.keys())}")
    #         for k, v in first.items():
    #             print(f"  {k}: {type(v).__name__} = {str(v)[:100]}")
    #     else:
    #         attrs = [a for a in dir(first) if not a.startswith('_')]
    #         print(f"DEBUG attrs: {attrs}")
    #         for a in attrs[:15]:
    #             try:
    #                 val = getattr(first, a)
    #                 if not callable(val):
    #                     print(f"  {a}: {type(val).__name__} = {str(val)[:100]}")
    #             except Exception:
    #                 pass

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"EVAL RESULTS — {experiment_name}")
    print("="*60)

    scores = []
    passed = 0
    failed = 0

    for result in all_results:
        question        = extract_question(result)
        score, comment  = extract_score(result)

        scores.append(score)
        status = "PASS" if score >= 0.7 else "FAIL"
        if score >= 0.7:
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] {question}...")
        print(f"         Score: {score:.1f}  {comment}")

    print()
    if scores:
        avg = sum(scores) / len(scores)
        print(f"  Total:   {len(scores)} questions")
        print(f"  Passed:  {passed} ({passed/len(scores)*100:.0f}%)")
        print(f"  Failed:  {failed} ({failed/len(scores)*100:.0f}%)")
        print(f"  Average: {avg:.2f} / 1.0")

    print()
    print(f"  Full results in LangSmith:")
    print(f"  https://smith.langchain.com/projects/{settings.LANGSMITH_PROJECT}")
    print("="*60)


if __name__ == "__main__":
    main()