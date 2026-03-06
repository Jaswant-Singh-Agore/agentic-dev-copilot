"""
Agent responsible for parsing error logs and identifying root causes.
Uses regex-based parsing, FAISS semantic search, and LLM reasoning.
"""

import logging

from mcp_server.tools import identify_root_cause, parse_error
from pipeline.indexer import search_codebase


logger = logging.getLogger(__name__)


class LogAnalyzerAgent:
    """Parses stack traces, finds similar code, and explains root causes in plain English."""

    def __init__(self) -> None:
        self.name = "LogAnalyzerAgent"

    def run(self, log_text: str) -> dict:
        """
        Run the full analysis pipeline for a raw error log or stack trace.

        Steps:
        1. Parse the error — extract type, message, file, line
        2. Search codebase for similar code to use as context
        3. Identify root cause using LLM

        Returns a dict with: parsed_error, similar_code, root_cause, confidence, agent.
        """
        logger.info("[%s] Starting analysis...", self.name)

        parsed = parse_error(log_text)
        logger.info(
            "[%s] Detected: %s at line %s",
            self.name,
            parsed.get("error_type"),
            parsed.get("line_number"),
        )

        search_query = " ".join(filter(None, [
            parsed.get("error_type"),
            parsed.get("error_message"),
            parsed.get("function_name"),
        ]))

        similar_chunks = search_codebase(search_query, top_k=2)
        context_code = None

        if similar_chunks:
            context_code = "\n\n# similar code from codebase\n".join(
                c["content"] for c in similar_chunks
            )
            logger.info("[%s] Found %d similar chunk(s).", self.name, len(similar_chunks))
        else:
            logger.info("[%s] No similar code found in codebase.", self.name)

        root_cause = identify_root_cause(parsed, context_code)

        # high confidence if we have both error type and location
        if parsed.get("error_type") and parsed.get("line_number"):
            confidence = "HIGH"
        elif parsed.get("error_type"):
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        logger.info("[%s] Analysis complete. Confidence: %s", self.name, confidence)

        return {
            "agent": self.name,
            "parsed_error": parsed,
            "similar_code": similar_chunks,
            "root_cause": root_cause,
            "confidence": confidence,
        }

    def format_report(self, analysis: dict) -> str:
        """Format the analysis result into a human-readable report for the Streamlit UI."""
        parsed = analysis.get("parsed_error", {})

        lines = [
            "=" * 60,
            "LOG ANALYSIS REPORT",
            "=" * 60,
            f"Error Type    : {parsed.get('error_type', 'Unknown')}",
            f"Error Message : {parsed.get('error_message', 'Unknown')}",
            f"File          : {parsed.get('file', 'Unknown')}",
            f"Line Number   : {parsed.get('line_number', 'Unknown')}",
            f"Function      : {parsed.get('function_name', 'Unknown')}",
            f"Confidence    : {analysis.get('confidence', 'Unknown')}",
            "",
            "ROOT CAUSE ANALYSIS",
            "-" * 40,
            analysis.get("root_cause", "No analysis available"),
        ]

        similar = analysis.get("similar_code", [])
        if similar:
            lines += ["", f"SIMILAR CODE FROM CODEBASE ({len(similar)} match(es))", "-" * 40]
            for i, chunk in enumerate(similar, 1):
                lines += [
                    f"\nMatch {i} — {chunk.get('file', 'unknown')} "
                    f"(lines {chunk.get('start_line')}-{chunk.get('end_line')}) "
                    f"[score: {chunk.get('similarity_score', 0):.3f}]",
                    chunk.get("content", ""),
                ]

        lines.append("=" * 60)
        return "\n".join(lines)