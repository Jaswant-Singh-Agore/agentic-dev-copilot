"""
Evaluation module for the Agentic Developer Copilot.
Measures quality of all 3 agents using custom metrics.

Run: python evaluation/eval.py
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from agents.code_fixer import CodeFixerAgent
from agents.log_analyzer import LogAnalyzerAgent
from agents.test_generator import TestGeneratorAgent


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


LOG_TEST_CASES = [
    {
        "id": "log_01",
        "description": "ZeroDivisionError in divide function",
        "log_text": """
Traceback (most recent call last):
  File "utils.py", line 10, in <module>
    result = divide(10, 0)
  File "utils.py", line 5, in divide
    return a / b
ZeroDivisionError: division by zero
""",
        "expected_error_type": "ZeroDivisionError",
        "expected_line": 5,
    },
    {
        "id": "log_02",
        "description": "KeyError in dictionary access",
        "log_text": """
Traceback (most recent call last):
  File "api_client.py", line 22, in get_user
    return data["user"]["name"]
KeyError: 'user'
""",
        "expected_error_type": "KeyError",
        "expected_line": 22,
    },
    {
        "id": "log_03",
        "description": "AttributeError on None object",
        "log_text": """
Traceback (most recent call last):
  File "processor.py", line 15, in process
    result = data.strip()
AttributeError: 'NoneType' object has no attribute 'strip'
""",
        "expected_error_type": "AttributeError",
        "expected_line": 15,
    },
]

FIX_TEST_CASES = [
    {
        "id": "fix_01",
        "description": "Fix ZeroDivisionError",
        "buggy_code": """
def divide(a, b):
    return a / b
""",
        "error_type": "ZeroDivisionError",
        "error_message": "division by zero",
        "expected_keywords_in_fix": ["if", "b", "0"],
    },
    {
        "id": "fix_02",
        "description": "Fix missing return statement",
        "buggy_code": """
def add(a, b):
    result = a + b
""",
        "error_type": "None",
        "error_message": "function returns None",
        "expected_keywords_in_fix": ["return"],
    },
]

TEST_GEN_CASES = [
    {
        "id": "test_01",
        "description": "Generate tests for calculator functions",
        "code": """
def add(a: int, b: int) -> int:
    return a + b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
""",
        "expected_min_tests": 2,
        "expected_function_names": ["add", "divide"],
    },
    {
        "id": "test_02",
        "description": "Generate tests for a class",
        "code": """
class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.items:
            raise IndexError("Stack is empty")
        return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0
""",
        "expected_min_tests": 2,
        "expected_function_names": ["Stack"],
    },
]


def score_log_analysis(result: dict, case: dict) -> dict:
    """Score a LogAnalyzerAgent result against expected values."""
    parsed = result.get("parsed_error", {})

    error_type_correct = int(parsed.get("error_type") == case["expected_error_type"])
    line_number_correct = int(parsed.get("line_number") == case["expected_line"])
    root_cause = result.get("root_cause", "")
    root_cause_present = int(len(root_cause) > 20)
    root_cause_length_score = min(len(root_cause) / 300, 1.0)
    confidence_valid = int(result.get("confidence") in ("HIGH", "MEDIUM", "LOW"))

    overall = round(
        error_type_correct * 0.30
        + line_number_correct * 0.20
        + root_cause_present * 0.30
        + root_cause_length_score * 0.10
        + confidence_valid * 0.10,
        3,
    )

    return {
        "error_type_correct": error_type_correct,
        "line_number_correct": line_number_correct,
        "root_cause_present": root_cause_present,
        "root_cause_length_score": round(root_cause_length_score, 3),
        "confidence_valid": confidence_valid,
        "overall": overall,
    }


def score_code_fix(result: dict, case: dict) -> dict:
    """Score a CodeFixerAgent result against expected values."""
    fix = result.get("fix", {})
    review = result.get("review", {})
    fixed_code = fix.get("fixed_code", "")

    fix_present = int(len(fixed_code) > 0)
    keywords = case.get("expected_keywords_in_fix", [])
    keywords_present = round(
        sum(1 for kw in keywords if kw in fixed_code) / len(keywords) if keywords else 0.0, 3
    )
    explanation_present = int(len(fix.get("explanation", "")) > 10)
    confidence_valid = int(fix.get("confidence") in ("HIGH", "MEDIUM", "LOW"))
    review_quality_valid = int(review.get("overall_quality", "UNKNOWN") != "UNKNOWN")

    overall = round(
        fix_present * 0.30
        + keywords_present * 0.25
        + explanation_present * 0.20
        + confidence_valid * 0.10
        + review_quality_valid * 0.15,
        3,
    )

    return {
        "fix_present": fix_present,
        "keywords_present": keywords_present,
        "explanation_present": explanation_present,
        "confidence_valid": confidence_valid,
        "review_quality_valid": review_quality_valid,
        "overall": overall,
    }


def score_test_generation(result: dict, case: dict) -> dict:
    """Score a TestGeneratorAgent result against expected values."""
    tests = result.get("tests", "")
    test_count = result.get("test_count", 0)

    tests_present = int(len(tests) > 0)
    min_tests_met = int(test_count >= case["expected_min_tests"])

    found_names = [f["name"] for f in result.get("functions_found", [])]
    expected = case["expected_function_names"]
    function_detection_score = round(
        sum(1 for name in expected if name in found_names) / len(expected) if expected else 0.0, 3
    )

    pytest_patterns = ["def test_", "assert", "pytest"]
    pytest_pattern_score = round(
        sum(1 for p in pytest_patterns if p in tests) / len(pytest_patterns), 3
    )
    coverage_notes_present = int(len(result.get("coverage_notes", "")) > 0)

    overall = round(
        tests_present * 0.25
        + min_tests_met * 0.20
        + function_detection_score * 0.20
        + pytest_pattern_score * 0.25
        + coverage_notes_present * 0.10,
        3,
    )

    return {
        "tests_present": tests_present,
        "min_tests_met": min_tests_met,
        "function_detection_score": function_detection_score,
        "pytest_pattern_score": pytest_pattern_score,
        "coverage_notes_present": coverage_notes_present,
        "overall": overall,
    }


def evaluate_log_analyzer() -> dict:
    print("\n" + "=" * 60)
    print("EVALUATING — LogAnalyzerAgent")
    print("=" * 60)

    agent = LogAnalyzerAgent()
    results = []

    for case in LOG_TEST_CASES:
        print(f"\n[{case['id']}] {case['description']}")
        start = time.time()
        result = agent.run(log_text=case["log_text"])
        elapsed = round(time.time() - start, 2)
        scores = score_log_analysis(result, case)

        parsed = result.get("parsed_error", {})
        print(f"  Score      : {scores['overall']:.3f} | Time: {elapsed}s")
        print(f"  Error type : {'pass' if scores['error_type_correct'] else 'fail'} "
              f"(expected: {case['expected_error_type']}, got: {parsed.get('error_type')})")
        print(f"  Line number: {'pass' if scores['line_number_correct'] else 'fail'} "
              f"(expected: {case['expected_line']}, got: {parsed.get('line_number')})")

        results.append({
            "case_id": case["id"],
            "description": case["description"],
            "scores": scores,
            "latency_s": elapsed,
        })

    avg_score = round(sum(r["scores"]["overall"] for r in results) / len(results), 3)
    print(f"\nLogAnalyzerAgent average score: {avg_score:.3f}")
    return {"agent": "LogAnalyzerAgent", "cases": results, "avg_score": avg_score}


def evaluate_code_fixer() -> dict:
    print("\n" + "=" * 60)
    print("EVALUATING — CodeFixerAgent")
    print("=" * 60)

    agent = CodeFixerAgent()
    results = []

    for case in FIX_TEST_CASES:
        print(f"\n[{case['id']}] {case['description']}")
        start = time.time()
        result = agent.run(
            buggy_code=case["buggy_code"],
            error_type=case["error_type"],
            error_message=case["error_message"],
        )
        elapsed = round(time.time() - start, 2)
        scores = score_code_fix(result, case)

        print(f"  Score      : {scores['overall']:.3f} | Time: {elapsed}s")
        print(f"  Fix present: {'pass' if scores['fix_present'] else 'fail'}")
        print(f"  Keywords   : {scores['keywords_present']:.0%}")
        print(f"  Confidence : {result.get('fix', {}).get('confidence', 'N/A')}")

        results.append({
            "case_id": case["id"],
            "description": case["description"],
            "scores": scores,
            "latency_s": elapsed,
        })

    avg_score = round(sum(r["scores"]["overall"] for r in results) / len(results), 3)
    print(f"\nCodeFixerAgent average score: {avg_score:.3f}")
    return {"agent": "CodeFixerAgent", "cases": results, "avg_score": avg_score}


def evaluate_test_generator() -> dict:
    print("\n" + "=" * 60)
    print("EVALUATING — TestGeneratorAgent")
    print("=" * 60)

    agent = TestGeneratorAgent()
    results = []

    for case in TEST_GEN_CASES:
        print(f"\n[{case['id']}] {case['description']}")
        start = time.time()
        result = agent.run(code=case["code"])
        elapsed = round(time.time() - start, 2)
        scores = score_test_generation(result, case)

        print(f"  Score           : {scores['overall']:.3f} | Time: {elapsed}s")
        print(f"  Tests present   : {'pass' if scores['tests_present'] else 'fail'}")
        print(f"  Min tests met   : {'pass' if scores['min_tests_met'] else 'fail'} "
              f"(count: {result.get('test_count', 0)})")
        print(f"  Function detect : {scores['function_detection_score']:.0%}")
        print(f"  Pytest patterns : {scores['pytest_pattern_score']:.0%}")

        results.append({
            "case_id": case["id"],
            "description": case["description"],
            "scores": scores,
            "latency_s": elapsed,
        })

    avg_score = round(sum(r["scores"]["overall"] for r in results) / len(results), 3)
    print(f"\nTestGeneratorAgent average score: {avg_score:.3f}")
    return {"agent": "TestGeneratorAgent", "cases": results, "avg_score": avg_score}


def save_report(results: list[dict]) -> str:
    """Save evaluation results to a JSON file."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "provider": "novita",
        "agents": results,
        "summary": {r["agent"]: r["avg_score"] for r in results},
    }
    output_path = Path("evaluation/eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Results saved to '%s'.", output_path)
    return str(output_path)


def print_summary(results: list[dict]) -> None:
    """Print a formatted summary table of all agent scores."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Agent':<25} {'Avg Score':>10} {'Cases':>8}")
    print("-" * 45)
    for r in results:
        print(f"{r['agent']:<25} {r['avg_score']:>10.3f} {len(r['cases']):>8}")
    print("-" * 45)
    overall = round(sum(r["avg_score"] for r in results) / len(results), 3)
    print(f"{'OVERALL':<25} {overall:>10.3f}")
    print("=" * 60)


if __name__ == "__main__":
    print(f"\nStarting evaluation — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = [
        evaluate_log_analyzer(),
        evaluate_code_fixer(),
        evaluate_test_generator(),
    ]

    print_summary(all_results)
    save_report(all_results)
    logger.info("Evaluation complete.")