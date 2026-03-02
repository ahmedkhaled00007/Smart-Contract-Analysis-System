"""
Evaluation Pipeline.
Provides basic metrics to evaluate RAG answer quality.
"""

from typing import List, Dict, Any
from config import logger


# ── Sample Test Cases ──────────────────────────────────────────────────────
SAMPLE_TEST_CASES = [
    {
        "question": "What is the termination clause?",
        "keywords": ["termination", "terminate", "cancel", "end"],
    },
    {
        "question": "Are there any penalties?",
        "keywords": ["penalty", "penalties", "fine", "fee", "breach"],
    },
    {
        "question": "What are the payment terms?",
        "keywords": ["payment", "pay", "invoice", "due", "amount"],
    },
]


def evaluate_answer(answer: str, ground_truth_keywords: List[str]) -> bool:
    """
    Checks if the answer contains any of the expected keywords.

    Args:
        answer: The generated answer text.
        ground_truth_keywords: List of keywords that should appear.

    Returns:
        True if at least one keyword is found.
    """
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in ground_truth_keywords)


def run_evaluation(chain, test_cases: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Runs a set of test cases through the RAG chain and evaluates them.

    Args:
        chain: The RAG chain to evaluate.
        test_cases: List of dicts with 'question' and 'keywords' keys.
                    Defaults to SAMPLE_TEST_CASES.

    Returns:
        Dict with 'results' (list of per-question results) and 'score'.
    """
    if test_cases is None:
        test_cases = SAMPLE_TEST_CASES

    results = []
    passed = 0

    for i, tc in enumerate(test_cases, 1):
        question = tc["question"]
        keywords = tc["keywords"]

        response = chain.invoke({"input": question})
        answer = response.get("answer", "")
        is_pass = evaluate_answer(answer, keywords)

        if is_pass:
            passed += 1

        result = {
            "question": question,
            "answer": answer[:200],  # truncate for display
            "expected_keywords": keywords,
            "passed": is_pass,
        }
        results.append(result)

        status = "✅ PASS" if is_pass else "❌ FAIL"
        logger.info(f"Test {i}/{len(test_cases)}: {status} — {question}")

    score = passed / len(test_cases) if test_cases else 0.0

    logger.info(f"Evaluation complete: {passed}/{len(test_cases)} passed ({score:.0%})")

    return {
        "results": results,
        "passed": passed,
        "total": len(test_cases),
        "score": score,
    }
