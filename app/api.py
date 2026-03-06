"""
FastAPI application for the Agentic Developer Copilot.
Exposes endpoints for log analysis, code fixing, review, test generation, and pipeline orchestration.
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from agents.log_analyzer import LogAnalyzerAgent
from agents.code_fixer import CodeFixerAgent
from agents.test_generator import TestGeneratorAgent
from pipeline.orchestrator import run_copilot
from pipeline.indexer import load_index
from dev_copilot_config import API_HOST, API_PORT


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic Developer Copilot",
    description="MCP-enabled multi-agent copilot for error analysis, code fixing, and test generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeLogRequest(BaseModel):
    log_text: str = Field(..., description="Raw stack trace or error log")

    @field_validator("log_text")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("log_text cannot be empty.")
        return v


class FixCodeRequest(BaseModel):
    buggy_code: str = Field(..., description="Broken Python code")
    error_type: str | None = Field(None, description="e.g. ZeroDivisionError")
    error_message: str | None = Field(None, description="Error message string")

    @field_validator("buggy_code")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("buggy_code cannot be empty.")
        return v


class ReviewCodeRequest(BaseModel):
    code: str = Field(..., description="Python code to review")

    @field_validator("code")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("code cannot be empty.")
        return v


class GenerateTestsRequest(BaseModel):
    code: str = Field(..., description="Python code to generate tests for")
    target_function: str | None = Field(None, description="Specific function to target")

    @field_validator("code")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("code cannot be empty.")
        return v


class FullPipelineRequest(BaseModel):
    task: str = Field(..., description="analyze | fix | test | full")
    log_text: str | None = Field(None, description="Error log or traceback")
    code: str | None = Field(None, description="Source code")
    target_function: str | None = Field(None, description="Target function for tests")


_VALID_TASKS = {"analyze", "fix", "test", "full"}


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok", "service": "Agentic Developer Copilot"}


@app.post("/analyze-log")
def analyze_log(request: AnalyzeLogRequest) -> dict:
    try:
        agent = LogAnalyzerAgent()
        result = agent.run(log_text=request.log_text)
        parsed = result.get("parsed_error", {})
        return {
            "agent": result.get("agent"),
            "error_type": parsed.get("error_type"),
            "error_message": parsed.get("error_message"),
            "file": parsed.get("file"),
            "line_number": parsed.get("line_number"),
            "function_name": parsed.get("function_name"),
            "root_cause": result.get("root_cause", ""),
            "confidence": result.get("confidence", "LOW"),
            "similar_code": result.get("similar_code", []),
            "report": agent.format_report(result),
        }
    except Exception as e:
        logger.exception("Error in /analyze-log")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fix-code")
def fix_code(request: FixCodeRequest) -> dict:
    try:
        agent = CodeFixerAgent()
        result = agent.run(
            buggy_code=request.buggy_code,
            error_type=request.error_type or "Unknown",
            error_message=request.error_message or "Unknown",
        )
        fix = result.get("fix", {})
        review = result.get("review", {})
        return {
            "agent": result.get("agent"),
            "fixed_code": fix.get("fixed_code", ""),
            "explanation": fix.get("explanation", ""),
            "confidence": fix.get("confidence", "LOW"),
            "functions_found": result.get("functions_found", []),
            "similar_code": result.get("similar_code", []),
            "review": {
                "overall_quality": review.get("overall_quality", "UNKNOWN"),
                "issues": review.get("issues", []),
                "suggestions": review.get("suggestions", []),
                "summary": review.get("summary", ""),
            },
            "report": agent.format_report(result),
        }
    except Exception as e:
        logger.exception("Error in /fix-code")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review-code")
def review_code(request: ReviewCodeRequest) -> dict:
    try:
        agent = CodeFixerAgent()
        result = agent.run_review_only(code=request.code)
        review = result.get("review", {})
        return {
            "agent": result.get("agent"),
            "functions_found": result.get("functions_found", []),
            "overall_quality": review.get("overall_quality", "UNKNOWN"),
            "issues": review.get("issues", []),
            "suggestions": review.get("suggestions", []),
            "summary": review.get("summary", ""),
        }
    except Exception as e:
        logger.exception("Error in /review-code")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-tests")
def generate_tests(request: GenerateTestsRequest) -> dict:
    try:
        agent = TestGeneratorAgent()
        result = agent.run(
            code=request.code,
            target_function=request.target_function,
        )
        return {
            "agent": result.get("agent"),
            "functions_found": result.get("functions_found", []),
            "target_function": result.get("target_function"),
            "tests": result.get("tests", ""),
            "test_count": result.get("test_count", 0),
            "coverage_notes": result.get("coverage_notes", ""),
            "report": agent.format_report(result),
        }
    except Exception as e:
        logger.exception("Error in /generate-tests")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-pipeline")
def run_pipeline(request: FullPipelineRequest) -> dict:
    if request.task not in _VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"task must be one of: {sorted(_VALID_TASKS)}",
        )
    if not request.log_text and not request.code:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: log_text, code.",
        )
    try:
        final_state = run_copilot(
            task=request.task,
            log_text=request.log_text,
            code=request.code,
            target_function=request.target_function,
        )
        return {
            "task": request.task,
            "steps_completed": final_state.get("steps_completed", []),
            "log_analysis": final_state.get("log_analysis"),
            "code_fix": final_state.get("code_fix"),
            "test_result": final_state.get("test_result"),
            "final_report": final_state.get("final_report", ""),
            "error": final_state.get("error"),
        }
    except Exception as e:
        logger.exception("Error in /run-pipeline")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index-status")
def index_status() -> dict:
    try:
        index, chunks = load_index()
        if index is None:
            return {
                "status": "not_built",
                "total_chunks": 0,
                "total_vectors": 0,
                "message": "No FAISS index found. Run build_index() first.",
            }
        return {
            "status": "ready",
            "total_chunks": len(chunks),
            "total_vectors": index.ntotal,
            "message": "Index is loaded and ready.",
        }
    except Exception as e:
        logger.exception("Error in /index-status")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host=API_HOST, port=API_PORT, reload=True)