"""
MCP tool implementations for the Dev Copilot pipeline.
Each function is a standalone tool: parse_error, identify_root_cause,
suggest_fix, review_code, generate_unit_tests, extract_functions.
"""

import ast
import json
import logging
import re

from huggingface_hub import InferenceClient

from dev_copilot_config import HF_API_TOKEN, LLM_MODEL, LLM_PROVIDER, MAX_LOG_LENGTH, MAX_TESTS_PER_FUNCTION


logger = logging.getLogger(__name__)

# single client instance reused across all tool calls
_client = InferenceClient(provider=LLM_PROVIDER, api_key=HF_API_TOKEN)


def _call_llm(prompt: str, system: str, max_tokens: int = 1024) -> str:
    """Send a prompt to the LLM and return the response text."""
    try:
        response = _client.chat_completion(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        return f"LLM Error: {e}"


def _parse_json_response(raw: str, fallback: dict) -> dict:
    """Strip markdown fences and parse JSON — returns fallback on failure."""
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Could not parse LLM response as JSON.")
        return fallback


def parse_error(log_text: str) -> dict:
    """
    Extract structured fields from a raw Python error log using regex.
    Truncates input to MAX_LOG_LENGTH before processing.
    """
    log_text = log_text[:MAX_LOG_LENGTH]

    result = {
        "error_type": None,
        "error_message": None,
        "file": None,
        "line_number": None,
        "function_name": None,
        "raw_log": log_text,
    }

    error_match = re.search(
        r"([A-Z][a-zA-Z]*Error|[A-Z][a-zA-Z]*Exception|[A-Z][a-zA-Z]*Warning):\s*(.+)",
        log_text,
    )
    if error_match:
        result["error_type"] = error_match.group(1)
        result["error_message"] = error_match.group(2).strip()

    file_match = re.search(r'File "([^"]+)",\s*line\s*(\d+)', log_text)
    if file_match:
        result["file"] = file_match.group(1)
        result["line_number"] = int(file_match.group(2))

    func_match = re.search(r"\bin\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\n", log_text)
    if func_match:
        result["function_name"] = func_match.group(1)

    return result


def identify_root_cause(parsed_error: dict, context_code: str | None = None) -> str:
    """Explain the root cause of a parsed Python error in plain English."""
    system = """You are an expert Python debugger. Explain the error clearly.

Format:
ROOT CAUSE: <one sentence>
EXPLANATION: <2-3 sentences>
COMMON TRIGGERS: <bullet list>"""

    prompt = (
        f"Analyze this Python error.\n\n"
        f"Error Type: {parsed_error.get('error_type')}\n"
        f"Error Message: {parsed_error.get('error_message')}\n"
        f"File: {parsed_error.get('file')}\n"
        f"Line: {parsed_error.get('line_number')}\n"
        f"Function: {parsed_error.get('function_name')}\n"
    )
    if context_code:
        prompt += f"\nRelevant Code:\n{context_code}"
    else:
        prompt += f"\nLog:\n{parsed_error.get('raw_log', '')[:1000]}"

    return _call_llm(prompt, system)


def suggest_fix(
    error_type: str,
    error_message: str,
    buggy_code: str,
    similar_code_context: str | None = None,
) -> dict:
    """Return a fixed version of buggy code along with an explanation and confidence level."""
    system = """You are an expert Python developer.

Return JSON:
{
  "fixed_code": "<corrected code>",
  "explanation": "<what changed>",
  "confidence": "<HIGH | MEDIUM | LOW>"
}
Return ONLY JSON."""

    prompt = (
        f"Fix this Python code.\n\n"
        f"ERROR TYPE: {error_type}\n"
        f"ERROR MESSAGE: {error_message}\n\n"
        f"BUGGY CODE:\n{buggy_code}"
    )
    if similar_code_context:
        prompt += f"\nREFERENCE CODE:\n{similar_code_context}"

    raw = _call_llm(prompt, system, max_tokens=1500)
    return _parse_json_response(raw, {
        "fixed_code": raw,
        "explanation": "Could not parse structured response.",
        "confidence": "LOW",
    })


def review_code(code: str) -> dict:
    """Run an automated code review and return structured findings."""
    system = """You are a senior Python code reviewer.

Return JSON:
{
  "overall_quality": "<GOOD | NEEDS_IMPROVEMENT | POOR>",
  "issues": [{"severity": "HIGH|MEDIUM|LOW", "line": <line|null>, "description": "<issue>"}],
  "suggestions": ["<suggestion>"],
  "summary": "<short summary>"
}
Return ONLY JSON."""

    raw = _call_llm(f"Review this Python code:\n{code}", system, max_tokens=1500)
    return _parse_json_response(raw, {
        "overall_quality": "UNKNOWN",
        "issues": [],
        "suggestions": [raw],
        "summary": "Could not parse structured response.",
    })


def generate_unit_tests(code: str, function_name: str | None = None) -> dict:
    """Generate pytest unit tests for the given code or a specific function."""
    system = f"""You are an expert Python test engineer.
Generate pytest tests (max {MAX_TESTS_PER_FUNCTION}).

Return ONLY this JSON with no triple quotes, no markdown, no extra formatting:
{{
  "test_code": "<complete pytest test file as a single-line string with \\n for newlines>",
  "test_count": <number>,
  "coverage_notes": "<coverage summary>"
}}
Return ONLY JSON."""

    target = f"for function {function_name}" if function_name else "for the code"
    raw = _call_llm(f"Generate unit tests {target}:\n{code}", system, max_tokens=2000)
    return _parse_json_response(raw, {
        "test_code": raw,
        "test_count": 0,
        "coverage_notes": "Could not parse structured response.",
    })


def extract_functions(code: str) -> list[dict]:
    """
    Parse Python source code with AST and return all function and class definitions.
    Returns an error entry if the code has a syntax error.
    """
    results = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                results.append({
                    "type": "function",
                    "name": node.name,
                    "line": node.lineno,
                    "docstring": (ast.get_docstring(node) or "")[:100],
                })
            elif isinstance(node, ast.ClassDef):
                results.append({
                    "type": "class",
                    "name": node.name,
                    "line": node.lineno,
                    "docstring": (ast.get_docstring(node) or "")[:100],
                })
    except SyntaxError as e:
        results.append({
            "type": "error",
            "name": "SyntaxError",
            "line": e.lineno,
            "docstring": str(e),
        })
    return results