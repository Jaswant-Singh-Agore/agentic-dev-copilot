"""
Agent responsible for fixing buggy Python code.
Uses AST extraction, FAISS semantic search, and LLM-based fix generation.
"""

import logging

from mcp_server.tools import extract_functions, review_code, suggest_fix
from pipeline.indexer import search_codebase


logger = logging.getLogger(__name__)


class CodeFixerAgent:
    """Searches the codebase for similar code, suggests a fix, and reviews it."""

    def __init__(self) -> None:
        self.name = "CodeFixerAgent"

    def run(
        self,
        buggy_code: str,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> dict:
        """
        Run the full fix pipeline for a buggy code snippet.

        Steps:
        1. Extract functions from buggy code using AST
        2. Search codebase for similar working code via FAISS
        3. Suggest a fix using LLM + similar code context
        4. Review the suggested fix

        Returns a dict with: functions_found, similar_code, fix, review, confidence, agent.
        """
        logger.info("[%s] Starting code fix pipeline...", self.name)

        functions = extract_functions(buggy_code)
        func_names = [f["name"] for f in functions if f["type"] in ("function", "class")]
        logger.info("[%s] Functions found: %s", self.name, func_names or "none")

        search_query = " ".join(filter(None, [
            error_type,
            error_message,
            " ".join(func_names),
            buggy_code[:200],
        ]))

        similar_chunks = search_codebase(search_query, top_k=3)
        similar_context = None

        if similar_chunks:
            similar_context = "\n\n# similar code\n".join(
                c["content"] for c in similar_chunks
            )
            logger.info("[%s] Found %d similar chunk(s).", self.name, len(similar_chunks))
        else:
            logger.info("[%s] No similar code found in codebase.", self.name)

        fix = suggest_fix(
            error_type=error_type or "Unknown",
            error_message=error_message or "Unknown",
            buggy_code=buggy_code,
            similar_code_context=similar_context,
        )
        logger.info("[%s] Fix confidence: %s", self.name, fix.get("confidence", "UNKNOWN"))

        fixed_code = fix.get("fixed_code", buggy_code)
        review = review_code(fixed_code)
        logger.info("[%s] Code quality: %s", self.name, review.get("overall_quality", "UNKNOWN"))

        return {
            "agent": self.name,
            "functions_found": functions,
            "similar_code": similar_chunks,
            "fix": fix,
            "review": review,
            "confidence": fix.get("confidence", "LOW"),
        }

    def run_review_only(self, code: str) -> dict:
        """Run a code review without generating a fix."""
        logger.info("[%s] Running code review only...", self.name)
        functions = extract_functions(code)
        review = review_code(code)
        logger.info("[%s] Review complete: %s", self.name, review.get("overall_quality", "UNKNOWN"))
        return {
            "agent": self.name,
            "functions_found": functions,
            "review": review,
        }

    def format_report(self, result: dict) -> str:
        """Format the fix result into a human-readable report for the Streamlit UI."""
        fix = result.get("fix", {})
        review = result.get("review", {})

        lines = [
            "=" * 60,
            "CODE FIX REPORT",
            "=" * 60,
            f"Confidence   : {result.get('confidence', 'UNKNOWN')}",
            f"Code Quality : {review.get('overall_quality', 'UNKNOWN')}",
            "",
            "EXPLANATION",
            "-" * 40,
            fix.get("explanation", "No explanation available"),
            "",
            "FIXED CODE",
            "-" * 40,
            fix.get("fixed_code", "No fix available"),
        ]

        issues = review.get("issues", [])
        if issues:
            lines += ["", "REMAINING ISSUES", "-" * 40]
            for issue in issues:
                severity = issue.get("severity", "?")
                line_no = issue.get("line", "?")
                desc = issue.get("description", "")
                lines.append(f"  [{severity}] Line {line_no}: {desc}")

        suggestions = review.get("suggestions", [])
        if suggestions:
            lines += ["", "SUGGESTIONS", "-" * 40]
            for s in suggestions:
                lines.append(f"  - {s}")

        similar = result.get("similar_code", [])
        if similar:
            lines += ["", f"SIMILAR CODE FROM CODEBASE ({len(similar)} match(es))", "-" * 40]
            for i, chunk in enumerate(similar, 1):
                lines.append(
                    f"\nMatch {i} — {chunk.get('file', 'unknown')} "
                    f"(lines {chunk.get('start_line')}-{chunk.get('end_line')}) "
                    f"[score: {chunk.get('similarity_score', 0):.3f}]"
                )

        lines.append("=" * 60)
        return "\n".join(lines)