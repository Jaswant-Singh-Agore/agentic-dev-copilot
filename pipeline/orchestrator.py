"""
LangGraph pipeline for the Agentic Developer Copilot.
Routes between LogAnalyzerAgent, CodeFixerAgent, and TestGeneratorAgent
based on the requested task.
"""

import logging
from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from agents.code_fixer import CodeFixerAgent
from agents.log_analyzer import LogAnalyzerAgent
from agents.test_generator import TestGeneratorAgent


logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    task: str
    log_text: str | None
    code: str | None
    target_function: str | None
    log_analysis: dict | None
    code_fix: dict | None
    test_result: dict | None
    final_report: str | None
    error: str | None
    steps_completed: list[str]


def _steps(state: AgentState, step: str) -> list[str]:
    return state.get("steps_completed", []) + [step]


def run_log_analyzer(state: AgentState) -> AgentState:
    logger.info("[Orchestrator] Running LogAnalyzerAgent...")
    try:
        result = LogAnalyzerAgent().run(state["log_text"])
        return {**state, "log_analysis": result, "steps_completed": _steps(state, "log_analyzer")}
    except Exception as e:
        logger.exception("[Orchestrator] LogAnalyzerAgent failed.")
        return {**state, "error": str(e), "steps_completed": _steps(state, "log_analyzer_failed")}


def run_code_fixer(state: AgentState) -> AgentState:
    logger.info("[Orchestrator] Running CodeFixerAgent...")
    try:
        log_analysis = state.get("log_analysis") or {}
        parsed_error = log_analysis.get("parsed_error", {})
        code = state.get("code") or parsed_error.get("raw_log", "")

        result = CodeFixerAgent().run(
            buggy_code=code,
            error_type=parsed_error.get("error_type"),
            error_message=parsed_error.get("error_message"),
        )
        return {**state, "code_fix": result, "steps_completed": _steps(state, "code_fixer")}
    except Exception as e:
        logger.exception("[Orchestrator] CodeFixerAgent failed.")
        return {**state, "error": str(e), "steps_completed": _steps(state, "code_fixer_failed")}


def run_test_generator(state: AgentState) -> AgentState:
    logger.info("[Orchestrator] Running TestGeneratorAgent...")
    try:
        agent = TestGeneratorAgent()
        fix = (state.get("code_fix") or {}).get("fix") or {}
        fixed_code = fix.get("fixed_code")
        original_code = state.get("code", "")

        if fixed_code and original_code:
            result = agent.run_for_fixed_code(original_code, fixed_code)
        else:
            result = agent.run(
                code=fixed_code or original_code,
                target_function=state.get("target_function"),
            )

        if result.get("tests"):
            agent.save_tests(result["tests"])

        return {**state, "test_result": result, "steps_completed": _steps(state, "test_generator")}
    except Exception as e:
        logger.exception("[Orchestrator] TestGeneratorAgent failed.")
        return {**state, "error": str(e), "steps_completed": _steps(state, "test_generator_failed")}


def compile_final_report(state: AgentState) -> AgentState:
    logger.info("[Orchestrator] Compiling final report...")
    sections = [
        "=" * 60,
        "DEVELOPER COPILOT — FINAL REPORT",
        "=" * 60,
        f"Task            : {state.get('task', 'unknown').upper()}",
        f"Steps Completed : {' → '.join(state.get('steps_completed', []))}",
        "",
    ]

    if state.get("log_analysis"):
        sections += [LogAnalyzerAgent().format_report(state["log_analysis"]), ""]
    if state.get("code_fix"):
        sections += [CodeFixerAgent().format_report(state["code_fix"]), ""]
    if state.get("test_result"):
        sections += [TestGeneratorAgent().format_report(state["test_result"]), ""]
    if state.get("error"):
        sections += ["-" * 40, "ERRORS ENCOUNTERED", "-" * 40, state["error"]]

    sections.append("=" * 60)
    return {**state, "final_report": "\n".join(sections)}


def _route_entry(state: AgentState) -> Literal["log_analyzer", "code_fixer", "test_generator"]:
    task = state.get("task", "full")
    has_log = bool((state.get("log_text") or "").strip())
    has_code = bool((state.get("code") or "").strip())

    if has_log and task in ("analyze", "full"):
        return "log_analyzer"
    if has_code and task == "test":
        return "test_generator"
    if has_code:
        return "code_fixer"
    return "log_analyzer"


def _route_after_log(state: AgentState) -> Literal["code_fixer", "compile_report"]:
    if state.get("task") in ("fix", "full"):
        return "code_fixer"
    return "compile_report"


def _route_after_fix(state: AgentState) -> Literal["test_generator", "compile_report"]:
    if state.get("task") == "full":
        return "test_generator"
    return "compile_report"


def _build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("log_analyzer", run_log_analyzer)
    graph.add_node("code_fixer", run_code_fixer)
    graph.add_node("test_generator", run_test_generator)
    graph.add_node("compile_report", compile_final_report)

    graph.set_conditional_entry_point(_route_entry, {
        "log_analyzer": "log_analyzer",
        "code_fixer": "code_fixer",
        "test_generator": "test_generator",
    })
    graph.add_conditional_edges("log_analyzer", _route_after_log, {
        "code_fixer": "code_fixer",
        "compile_report": "compile_report",
    })
    graph.add_conditional_edges("code_fixer", _route_after_fix, {
        "test_generator": "test_generator",
        "compile_report": "compile_report",
    })
    graph.add_edge("test_generator", "compile_report")
    graph.add_edge("compile_report", END)

    return graph.compile()


def run_copilot(
    task: str,
    log_text: str | None = None,
    code: str | None = None,
    target_function: str | None = None,
) -> dict:
    """Run the full copilot pipeline. task must be one of: analyze, fix, test, full."""
    logger.info("[Orchestrator] Starting — task: %s", task.upper())

    initial_state: AgentState = {
        "task": task,
        "log_text": log_text or "",
        "code": code or "",
        "target_function": target_function,
        "log_analysis": None,
        "code_fix": None,
        "test_result": None,
        "final_report": None,
        "error": None,
        "steps_completed": [],
    }

    final_state = _build_graph().invoke(initial_state)
    logger.info("[Orchestrator] Done — steps: %s", " → ".join(final_state.get("steps_completed", [])))
    return final_state