"""
Streamlit frontend for the Agentic Developer Copilot.
Requires the FastAPI backend running at API_BASE_URL.
"""

from __future__ import annotations

import requests
import streamlit as st
from requests.exceptions import ConnectionError, RequestException, Timeout

API_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 60
HEALTH_CHECK_TIMEOUT = 3

SEVERITY_COLORS = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
QUALITY_ICONS = {"GOOD": "🟢", "NEEDS_IMPROVEMENT": "🟡", "POOR": "🔴"}
CONFIDENCE_ICONS = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}

st.set_page_config(
    page_title="Agentic Developer Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


class APIClient:
    """Wraps all backend calls and owns the requests.Session lifetime."""

    def __init__(self, base_url: str, timeout: int = REQUEST_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def is_healthy(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=HEALTH_CHECK_TIMEOUT)
            return r.status_code == 200
        except (ConnectionError, Timeout, RequestException):
            return False

    def analyze_log(self, log_text: str) -> tuple[dict, int]:
        return self._post("/analyze-log", {"log_text": log_text})

    def fix_code(self, buggy_code: str, error_type: str | None, error_message: str | None) -> tuple[dict, int]:
        return self._post("/fix-code", {
            "buggy_code": buggy_code,
            "error_type": error_type,
            "error_message": error_message,
        })

    def review_code(self, code: str) -> tuple[dict, int]:
        return self._post("/review-code", {"code": code})

    def generate_tests(self, code: str, target_function: str | None = None) -> tuple[dict, int]:
        return self._post("/generate-tests", {"code": code, "target_function": target_function})

    def _post(self, endpoint: str, payload: dict) -> tuple[dict, int]:
        response = self.session.post(f"{self.base_url}{endpoint}", json=payload, timeout=self.timeout)
        return response.json(), response.status_code


_api = APIClient(base_url=API_BASE_URL)


def _show_api_error(data: dict, status_code: int) -> None:
    st.error(f"API Error {status_code}: {data.get('detail', 'Unknown error')}")


def _show_expanders(data: dict) -> None:
    if report := data.get("report", ""):
        with st.expander("Full Report"):
            st.text(report)
    with st.expander("Raw JSON"):
        st.json(data)


def _render_code_review(review: dict) -> None:
    quality = review.get("overall_quality", "UNKNOWN")
    st.markdown(f"**Code Quality:** {QUALITY_ICONS.get(quality, '⚪')} `{quality}`")

    if summary := review.get("summary"):
        st.info(summary)

    issues: list[dict] = review.get("issues", [])
    if issues:
        st.markdown("**Issues Found:**")
        for issue in issues:
            sev = issue.get("severity", "?")
            st.markdown(
                f"- `[{sev}]` Line {issue.get('line', '?')}: {issue.get('description', '')}"
            )

    suggestions: list[str] = review.get("suggestions", [])
    if suggestions:
        st.markdown("**Suggestions:**")
        for s in suggestions:
            st.markdown(f"- {s}")


def _error_analyzer_tab() -> None:
    st.header("Error Analyzer")
    st.caption("Paste a Python error log or stack trace — the agent explains the root cause in plain English.")

    log_text = st.text_area(
        "Error Log / Stack Trace",
        height=250,
        key="log_input",
        placeholder="Traceback (most recent call last):\n  ...",
    )

    if not st.button("Analyze", type="primary", key="analyze_btn"):
        return

    if not log_text.strip():
        st.warning("Please paste an error log first.")
        return

    with st.spinner("Analyzing error..."):
        try:
            data, status_code = _api.analyze_log(log_text)
        except RequestException as e:
            st.error(f"Network error: {e}")
            return

    if status_code != 200:
        _show_api_error(data, status_code)
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Error Type", data.get("error_type", "Unknown"))
    col2.metric("Line Number", data.get("line_number", "Unknown"))
    col3.metric("Function", data.get("function_name", "Unknown"))
    col4.metric("Confidence", data.get("confidence", "Unknown"))

    if error_message := data.get("error_message"):
        st.code(error_message, language="text")

    st.subheader("Root Cause Analysis")
    st.markdown(data.get("root_cause", "No analysis available."))

    similar: list[dict] = data.get("similar_code", [])
    if similar:
        st.subheader(f"Similar Code Found ({len(similar)} match(es))")
        for i, chunk in enumerate(similar, start=1):
            with st.expander(f"Match {i} — {chunk.get('file', 'unknown')}"):
                st.code(chunk.get("content", ""), language="python")

    _show_expanders(data)


def _code_fixer_tab() -> None:
    st.header("Code Fixer")
    st.caption("Paste broken Python code — the agent suggests a fix and reviews it.")

    left_col, right_col = st.columns(2)
    with left_col:
        buggy_code = st.text_area(
            "Buggy Code",
            height=300,
            key="buggy_code_input",
            placeholder="# Paste your broken Python code here",
        )
    with right_col:
        error_type = st.text_input("Error Type (optional)", key="error_type_input", placeholder="e.g. TypeError")
        error_message = st.text_input("Error Message (optional)", key="error_message_input",
                                      placeholder="e.g. unsupported operand type(s)")
        review_only = st.checkbox("Review only (no fix)", key="review_only_cb",
                                  help="Run a code review without generating a fix.")

    if not st.button("Review Code" if review_only else "Fix Code", type="primary", key="fix_btn"):
        return

    if not buggy_code.strip():
        st.warning("Please paste some code first.")
        return

    with st.spinner("Reviewing..." if review_only else "Fixing..."):
        try:
            if review_only:
                data, status_code = _api.review_code(buggy_code)
            else:
                data, status_code = _api.fix_code(buggy_code, error_type or None, error_message or None)
        except RequestException as e:
            st.error(f"Network error: {e}")
            return

    if status_code != 200:
        _show_api_error(data, status_code)
        return

    if review_only:
        st.subheader("Code Review")
        _render_code_review(data)
        _show_expanders(data)
        return

    confidence = data.get("confidence", "LOW")
    st.markdown(f"**Fix Confidence:** {CONFIDENCE_ICONS.get(confidence, '⚪')} `{confidence}`")

    before_col, after_col = st.columns(2)
    with before_col:
        st.markdown("**Original (Buggy)**")
        st.code(buggy_code, language="python")
    with after_col:
        st.markdown("**Fixed**")
        st.code(data.get("fixed_code", ""), language="python")

    st.subheader("Explanation")
    st.markdown(data.get("explanation", "No explanation available."))

    st.subheader("Code Review of Fix")
    _render_code_review(data.get("review", {}))
    _show_expanders(data)


def _test_generator_tab() -> None:
    st.header("Test Generator")
    st.caption("Paste Python code — the agent generates ready-to-run `pytest` unit tests.")

    code_col, options_col = st.columns([3, 1])
    with code_col:
        test_code = st.text_area(
            "Python Code",
            height=300,
            key="test_code_input",
            placeholder="# Paste the Python code you want tests for",
        )
    with options_col:
        target_function = st.text_input(
            "Target Function (optional)",
            key="target_fn_input",
            placeholder="e.g. my_function",
            help="Leave blank to generate tests for all functions.",
        )

    if not st.button("Generate Tests", type="primary", key="test_btn"):
        return

    if not test_code.strip():
        st.warning("Please paste some code first.")
        return

    with st.spinner("Generating tests..."):
        try:
            data, status_code = _api.generate_tests(test_code, target_function or None)
        except RequestException as e:
            st.error(f"Network error: {e}")
            return

    if status_code != 200:
        _show_api_error(data, status_code)
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Tests Generated", data.get("test_count", 0))
    col2.metric("Functions Found", len(data.get("functions_found", [])))
    col3.metric("Target Function", data.get("target_function") or "All")

    if coverage_notes := data.get("coverage_notes"):
        st.subheader("Coverage Notes")
        st.info(coverage_notes)

    generated_tests = data.get("tests", "")
    if generated_tests:
        st.subheader("Generated Tests")
        st.code(generated_tests, language="python")
        st.download_button(
            label="Download test_generated.py",
            data=generated_tests,
            file_name="test_generated.py",
            mime="text/x-python",
        )
    else:
        st.warning("No tests were generated. Try providing a more complete code snippet.")

    _show_expanders(data)


def main() -> None:
    st.title("Agentic Developer Copilot")
    st.caption("MCP-enabled multi-agent assistant for error analysis, code fixing, and test generation.")

    if not _api.is_healthy():
        st.error("Backend unavailable. Start it with: `python app/api.py`")
        st.stop()

    st.success(f"API connected — {API_BASE_URL}")

    tab_analyzer, tab_fixer, tab_tests = st.tabs(["Error Analyzer", "Code Fixer", "Test Generator"])

    with tab_analyzer:
        _error_analyzer_tab()
    with tab_fixer:
        _code_fixer_tab()
    with tab_tests:
        _test_generator_tab()


if __name__ == "__main__":
    main()