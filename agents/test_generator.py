"""
Agent responsible for generating pytest unit tests from Python code.
Uses AST extraction to identify functions and LLM to generate test cases.
"""

import logging
from pathlib import Path

from mcp_server.tools import extract_functions, generate_unit_tests


logger = logging.getLogger(__name__)


class TestGeneratorAgent:
    """Extracts functions via AST and generates pytest unit tests using an LLM."""

    def __init__(self) -> None:
        self.name = "TestGeneratorAgent"

    def run(self, code: str, target_function: str | None = None) -> dict:
        """
        Generate pytest tests for the given code or a specific function.
        If target_function is provided but not found in the code, tests are
        generated for all functions instead.
        """
        logger.info("[%s] Starting test generation...", self.name)

        functions = extract_functions(code)
        func_names = [f["name"] for f in functions if f["type"] in ("function", "class")]

        # only use target_function if it actually exists in the code
        selected = target_function if target_function in func_names else None

        result = generate_unit_tests(code, function_name=selected)

        return {
            "agent": self.name,
            "functions_found": functions,
            "target_function": selected,
            "tests": result.get("test_code", ""),
            "test_count": result.get("test_count", 0),
            "coverage_notes": result.get("coverage_notes", ""),
        }

    def run_for_fixed_code(self, original_code: str, fixed_code: str) -> dict:
        """Generate tests for fixed code, including original as commented context."""
        commented_original = "\n".join(f"# {line}" for line in original_code.splitlines())
        combined = (
            f"# FIXED CODE (generate tests for this):\n{fixed_code}\n\n"
            f"# ORIGINAL BUGGY CODE (for context only):\n{commented_original}\n"
        )
        return self.run(combined)

    def save_tests(self, tests: str, output_path: str = "tests/generated_tests.py") -> str:
        """Write generated test code to a file, creating parent directories if needed."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tests, encoding="utf-8")
        logger.info("[%s] Tests saved to '%s'.", self.name, output_path)
        return output_path

    def format_report(self, result: dict) -> str:
        """Format the test generation result into a human-readable report."""
        lines = [
            "=" * 60,
            "TEST GENERATION REPORT",
            "=" * 60,
            f"Tests Generated : {result.get('test_count', 0)}",
            f"Target Function : {result.get('target_function', 'All functions')}",
            "",
            "COVERAGE NOTES",
            "-" * 40,
            result.get("coverage_notes", "No coverage notes available"),
        ]

        functions = result.get("functions_found", [])
        if functions:
            lines += ["", "FUNCTIONS DETECTED (AST)", "-" * 40]
            for f in functions:
                kind = f["type"].upper()
                lines.append(f"  {kind:10} line {f['line']:>4} — {f['name']}()")

        tests = result.get("tests", "")
        if tests:
            lines += ["", "GENERATED PYTEST CODE", "-" * 40, tests]
        else:
            lines.append("\nNo tests generated.")

        lines.append("=" * 60)
        return "\n".join(lines)