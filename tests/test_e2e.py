# tests/test_e2e.py
# End-to-end tests — direct agent tests + API integration tests
# Run: pytest tests/test_e2e.py -v
# Note: Requires HF_TOKEN in .env and API running on port 8000 for API tests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest

import requests

from agents.log_analyzer   import LogAnalyzerAgent
from agents.code_fixer     import CodeFixerAgent
from agents.test_generator import TestGeneratorAgent
from pipeline.orchestrator import run_copilot
from pipeline.indexer      import build_index, load_index, search_codebase

API_BASE = "http://127.0.0.1:8000"


# ── Shared fixtures ────────────────────────────────────────────────────────────

SAMPLE_TRACEBACK = """
Traceback (most recent call last):
  File "utils.py", line 10, in <module>
    result = divide(10, 0)
  File "utils.py", line 5, in divide
    return a / b
ZeroDivisionError: division by zero
"""

SAMPLE_BUGGY_CODE = """
def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
"""

SAMPLE_GOOD_CODE = """
def add(a: int, b: int) -> int:
    return a + b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

class Calculator:
    def multiply(self, a: int, b: int) -> int:
        return a * b
"""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — FAISS Index Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestFAISSIndex:

    def test_build_index_returns_success(self):
        """Index builds without error and returns success status."""
        print("\n[Test] Building FAISS index...")
        result = build_index()
        assert result["status"] in ("success", "warning"), (
            f"Expected success or warning, got: {result['status']}"
        )
        print(f"[Test] Index built — {result.get('chunks_indexed', 0)} chunks.")

    def test_load_index_returns_tuple(self):
        """load_index returns (index, chunks) tuple."""
        print("\n[Test] Loading FAISS index...")
        index, chunks = load_index()
        assert isinstance(chunks, list), "chunks should be a list"
        print(f"[Test] Loaded {len(chunks)} chunks.")

    def test_search_codebase_returns_list(self):
        """search_codebase returns a list of results."""
        print("\n[Test] Searching codebase...")
        results = search_codebase("ZeroDivisionError division by zero", top_k=2)
        assert isinstance(results, list), "Results should be a list"
        print(f"[Test] Found {len(results)} result(s).")

    def test_search_result_has_required_keys(self):
        """Each search result contains expected keys."""
        results = search_codebase("function definition", top_k=1)
        if results:
            chunk = results[0]
            for key in ("content", "file", "start_line", "end_line", "similarity_score"):
                assert key in chunk, f"Missing key: {key}"
            print(f"[Test] Result keys validated: {list(chunk.keys())}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LogAnalyzerAgent Direct Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestLogAnalyzerAgent:

    def setup_method(self):
        self.agent = LogAnalyzerAgent()

    def test_run_returns_dict(self):
        """Agent returns a dict."""
        print("\n[Test] Running LogAnalyzerAgent...")
        result = self.agent.run(SAMPLE_TRACEBACK)
        assert isinstance(result, dict), "Result should be a dict"

    def test_run_has_required_keys(self):
        """Result contains all expected keys."""
        result = self.agent.run(SAMPLE_TRACEBACK)
        for key in ("agent", "parsed_error", "similar_code", "root_cause", "confidence"):
            assert key in result, f"Missing key: {key}"

    def test_parsed_error_detects_type(self):
        """Correctly parses ZeroDivisionError from traceback."""
        result = self.agent.run(SAMPLE_TRACEBACK)
        parsed = result["parsed_error"]
        assert parsed.get("error_type") == "ZeroDivisionError", (
            f"Expected ZeroDivisionError, got: {parsed.get('error_type')}"
        )
        print(f"[Test] Detected error type: {parsed.get('error_type')}")

    def test_parsed_error_detects_line_number(self):
        """Correctly parses line number from traceback."""
        result = self.agent.run(SAMPLE_TRACEBACK)
        parsed = result["parsed_error"]
        assert parsed.get("line_number") is not None, "Line number should be parsed"
        print(f"[Test] Detected line number: {parsed.get('line_number')}")

    def test_root_cause_is_non_empty_string(self):
        """LLM returns a non-empty root cause explanation."""
        result = self.agent.run(SAMPLE_TRACEBACK)
        root_cause = result.get("root_cause", "")
        assert isinstance(root_cause, str), "root_cause should be a string"
        assert len(root_cause) > 10, "root_cause should be a meaningful string"
        print(f"[Test] Root cause length: {len(root_cause)} chars")

    def test_confidence_is_valid_value(self):
        """Confidence is one of HIGH, MEDIUM, LOW."""
        result = self.agent.run(SAMPLE_TRACEBACK)
        assert result.get("confidence") in ("HIGH", "MEDIUM", "LOW"), (
            f"Unexpected confidence: {result.get('confidence')}"
        )

    def test_format_report_returns_string(self):
        """format_report returns a non-empty string."""
        result = self.agent.run(SAMPLE_TRACEBACK)
        report = self.agent.format_report(result)
        assert isinstance(report, str)
        assert len(report) > 0
        print(f"[Test] Report length: {len(report)} chars")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CodeFixerAgent Direct Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCodeFixerAgent:

    def setup_method(self):
        self.agent = CodeFixerAgent()

    def test_run_returns_dict(self):
        """Agent returns a dict."""
        print("\n[Test] Running CodeFixerAgent...")
        result = self.agent.run(
            buggy_code    = SAMPLE_BUGGY_CODE,
            error_type    = "ZeroDivisionError",
            error_message = "division by zero",
        )
        assert isinstance(result, dict)

    def test_run_has_required_keys(self):
        """Result contains all expected keys."""
        result = self.agent.run(buggy_code=SAMPLE_BUGGY_CODE)
        for key in ("agent", "functions_found", "similar_code", "fix", "review", "confidence"):
            assert key in result, f"Missing key: {key}"

    def test_fix_is_dict_with_required_keys(self):
        """fix field is a dict with fixed_code, explanation, confidence."""
        result = self.agent.run(buggy_code=SAMPLE_BUGGY_CODE)
        fix = result.get("fix", {})
        assert isinstance(fix, dict), "fix should be a dict"
        for key in ("fixed_code", "explanation", "confidence"):
            assert key in fix, f"Missing key in fix: {key}"
        print(f"[Test] Fix confidence: {fix.get('confidence')}")

    def test_fixed_code_is_non_empty(self):
        """LLM returns non-empty fixed code."""
        result = self.agent.run(buggy_code=SAMPLE_BUGGY_CODE)
        fixed_code = result["fix"].get("fixed_code", "")
        assert len(fixed_code) > 0, "fixed_code should not be empty"
        print(f"[Test] Fixed code length: {len(fixed_code)} chars")

    def test_review_has_quality_field(self):
        """Review contains overall_quality field."""
        result = self.agent.run(buggy_code=SAMPLE_BUGGY_CODE)
        review = result.get("review", {})
        assert "overall_quality" in review, "review missing overall_quality"
        print(f"[Test] Code quality: {review.get('overall_quality')}")

    def test_run_review_only_returns_dict(self):
        """run_review_only returns dict with review key."""
        result = self.agent.run_review_only(code=SAMPLE_GOOD_CODE)
        assert isinstance(result, dict)
        assert "review" in result
        print(f"[Test] Review quality: {result['review'].get('overall_quality')}")

    def test_format_report_returns_string(self):
        """format_report returns a non-empty string."""
        result = self.agent.run(buggy_code=SAMPLE_BUGGY_CODE)
        report = self.agent.format_report(result)
        assert isinstance(report, str)
        assert len(report) > 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TestGeneratorAgent Direct Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTestGeneratorAgent:

    def setup_method(self):
        self.agent = TestGeneratorAgent()

    def test_run_returns_dict(self):
        """Agent returns a dict."""
        print("\n[Test] Running TestGeneratorAgent...")
        result = self.agent.run(code=SAMPLE_GOOD_CODE)
        assert isinstance(result, dict)

    def test_run_has_required_keys(self):
        """Result contains all expected keys."""
        result = self.agent.run(code=SAMPLE_GOOD_CODE)
        for key in ("agent", "functions_found", "target_function", "tests", "test_count", "coverage_notes"):
            assert key in result, f"Missing key: {key}"

    def test_functions_found_is_list(self):
        """functions_found is a non-empty list."""
        result = self.agent.run(code=SAMPLE_GOOD_CODE)
        funcs = result.get("functions_found", [])
        assert isinstance(funcs, list)
        assert len(funcs) > 0, "Should detect at least one function"
        print(f"[Test] Functions found: {[f['name'] for f in funcs]}")

    def test_detects_correct_function_names(self):
        """AST correctly extracts add, divide, Calculator."""
        result = self.agent.run(code=SAMPLE_GOOD_CODE)
        names = [f["name"] for f in result["functions_found"]]
        for expected in ("add", "divide", "Calculator"):
            assert expected in names, f"Expected '{expected}' in {names}"

    def test_tests_is_non_empty_string(self):
        """LLM returns non-empty test code."""
        result = self.agent.run(code=SAMPLE_GOOD_CODE)
        assert isinstance(result.get("tests"), str)
        assert len(result["tests"]) > 0, "tests should not be empty"
        print(f"[Test] Generated tests length: {len(result['tests'])} chars")

    def test_target_function_filtering(self):
        """When target_function is set, it is reflected in result."""
        result = self.agent.run(code=SAMPLE_GOOD_CODE, target_function="add")
        assert result.get("target_function") == "add", (
            f"Expected target_function='add', got: {result.get('target_function')}"
        )

    def test_run_for_fixed_code_returns_dict(self):
        """run_for_fixed_code returns a valid result dict."""
        result = self.agent.run_for_fixed_code(
            original_code = SAMPLE_BUGGY_CODE,
            fixed_code    = SAMPLE_GOOD_CODE,
        )
        assert isinstance(result, dict)
        assert "tests" in result

    def test_format_report_returns_string(self):
        """format_report returns a non-empty string."""
        result = self.agent.run(code=SAMPLE_GOOD_CODE)
        report = self.agent.format_report(result)
        assert isinstance(report, str)
        assert len(report) > 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LangGraph Orchestrator Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestrator:

    def test_run_copilot_analyze_task(self):
        """Orchestrator runs analyze task and returns log_analysis."""
        print("\n[Test] Running orchestrator — task: analyze...")
        state = run_copilot(task="analyze", log_text=SAMPLE_TRACEBACK)
        assert state.get("log_analysis") is not None, "log_analysis should be populated"
        assert "log_analyzer" in state.get("steps_completed", [])
        print(f"[Test] Steps: {state.get('steps_completed')}")

    def test_run_copilot_fix_task(self):
        """Orchestrator runs fix task and returns code_fix."""
        print("\n[Test] Running orchestrator — task: fix...")
        state = run_copilot(task="fix", code=SAMPLE_BUGGY_CODE)
        assert state.get("code_fix") is not None, "code_fix should be populated"
        assert "code_fixer" in state.get("steps_completed", [])

    def test_run_copilot_test_task(self):
        """Orchestrator runs test task and returns test_result."""
        print("\n[Test] Running orchestrator — task: test...")
        state = run_copilot(task="test", code=SAMPLE_GOOD_CODE)
        assert state.get("test_result") is not None, "test_result should be populated"
        assert "test_generator" in state.get("steps_completed", [])

    def test_run_copilot_full_pipeline(self):
        """Orchestrator runs full pipeline — all 3 agents fire."""
        print("\n[Test] Running orchestrator — task: full...")
        state = run_copilot(
            task     = "full",
            log_text = SAMPLE_TRACEBACK,
            code     = SAMPLE_BUGGY_CODE,
        )
        steps = state.get("steps_completed", [])
        assert "log_analyzer"   in steps, "log_analyzer should have run"
        assert "code_fixer"     in steps, "code_fixer should have run"
        assert "test_generator" in steps, "test_generator should have run"
        assert state.get("final_report") is not None
        print(f"[Test] All steps completed: {steps}")

    def test_final_report_is_non_empty(self):
        """Final report is a non-empty string."""
        state = run_copilot(task="analyze", log_text=SAMPLE_TRACEBACK)
        report = state.get("final_report", "")
        assert isinstance(report, str)
        assert len(report) > 0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — API Integration Tests (requires API running on port 8000)
# ══════════════════════════════════════════════════════════════════════════════

def api_available() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

skip_if_no_api = pytest.mark.skipif(
    not api_available(),
    reason="API not running on port 8000 — start with: python app/api.py",
)


class TestAPI:

    @skip_if_no_api
    def test_health_endpoint(self):
        """GET /health returns ok status."""
        r = requests.get(f"{API_BASE}/health")
        assert r.status_code == 200
        assert r.json().get("status") == "ok"
        print("\n[Test] /health ✅")

    @skip_if_no_api
    def test_index_status_endpoint(self):
        """GET /index-status returns valid response."""
        r = requests.get(f"{API_BASE}/index-status")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "total_chunks" in data
        print(f"\n[Test] /index-status → {data.get('status')}, chunks: {data.get('total_chunks')}")

    @skip_if_no_api
    def test_analyze_log_endpoint(self):
        """POST /analyze-log returns structured analysis."""
        r = requests.post(
            f"{API_BASE}/analyze-log",
            json    = {"log_text": SAMPLE_TRACEBACK},
            timeout = 60,
        )
        assert r.status_code == 200
        data = r.json()
        assert data.get("error_type") == "ZeroDivisionError"
        assert "root_cause" in data
        assert len(data.get("root_cause", "")) > 0
        print(f"\n[Test] /analyze-log ✅ — error_type: {data.get('error_type')}")

    @skip_if_no_api
    def test_analyze_log_empty_input(self):
        """POST /analyze-log with empty input returns 400."""
        r = requests.post(
            f"{API_BASE}/analyze-log",
            json = {"log_text": ""},
        )
        assert r.status_code in (400, 422)
        print("\n[Test] /analyze-log empty input → 400 ✅")

    @skip_if_no_api
    def test_fix_code_endpoint(self):
        """POST /fix-code returns fixed code."""
        r = requests.post(
            f"{API_BASE}/fix-code",
            json    = {
                "buggy_code"   : SAMPLE_BUGGY_CODE,
                "error_type"   : "ZeroDivisionError",
                "error_message": "division by zero",
            },
            timeout = 60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "fixed_code"  in data
        assert "explanation" in data
        assert "confidence"  in data
        assert len(data.get("fixed_code", "")) > 0
        print(f"\n[Test] /fix-code ✅ — confidence: {data.get('confidence')}")

    @skip_if_no_api
    def test_review_code_endpoint(self):
        """POST /review-code returns quality assessment."""
        r = requests.post(
            f"{API_BASE}/review-code",
            json    = {"code": SAMPLE_GOOD_CODE},
            timeout = 60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "overall_quality" in data
        assert data.get("overall_quality") in ("GOOD", "NEEDS_IMPROVEMENT", "POOR", "UNKNOWN")
        print(f"\n[Test] /review-code ✅ — quality: {data.get('overall_quality')}")

    @skip_if_no_api
    def test_generate_tests_endpoint(self):
        """POST /generate-tests returns pytest code."""
        r = requests.post(
            f"{API_BASE}/generate-tests",
            json    = {"code": SAMPLE_GOOD_CODE},
            timeout = 60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "tests"      in data
        assert "test_count" in data
        assert len(data.get("tests", "")) > 0
        print(f"\n[Test] /generate-tests ✅ — test_count: {data.get('test_count')}")

    @skip_if_no_api
    def test_run_pipeline_full(self):
        """POST /run-pipeline with task=full runs all agents."""
        r = requests.post(
            f"{API_BASE}/run-pipeline",
            json    = {
                "task"    : "full",
                "log_text": SAMPLE_TRACEBACK,
                "code"    : SAMPLE_BUGGY_CODE,
            },
            timeout = 120,
        )
        assert r.status_code == 200
        data = r.json()
        steps = data.get("steps_completed", [])
        assert "log_analyzer"   in steps
        assert "code_fixer"     in steps
        assert "test_generator" in steps
        assert len(data.get("final_report", "")) > 0
        print(f"\n[Test] /run-pipeline full ✅ — steps: {steps}")

    @skip_if_no_api
    def test_run_pipeline_invalid_task(self):
        """POST /run-pipeline with invalid task returns 400."""
        r = requests.post(
            f"{API_BASE}/run-pipeline",
            json = {"task": "invalid_task", "code": SAMPLE_GOOD_CODE},
        )
        assert r.status_code == 400
        print("\n[Test] /run-pipeline invalid task → 400 ✅")

    @skip_if_no_api
    def test_run_pipeline_no_input(self):
        """POST /run-pipeline with no log or code returns 400."""
        r = requests.post(
            f"{API_BASE}/run-pipeline",
            json = {"task": "full"},
        )
        assert r.status_code == 400
        print("\n[Test] /run-pipeline no input → 400 ✅")