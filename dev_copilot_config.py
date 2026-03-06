"""
Centralised configuration loaded from environment variables.
Fails fast if any required variable is missing.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Read a required env variable — raises early if missing."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val


# HuggingFace
HF_API_TOKEN = _require("HF_TOKEN")

# models
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLM_PROVIDER = "novita"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# storage
FAISS_INDEX_PATH = "data/faiss_index"
SAMPLE_CODE_PATH = "data/sample_code"

# agent settings
MAX_AGENT_ITERATIONS = 10
CONFIDENCE_THRESHOLD = 0.75
MAX_LOG_LENGTH = 5000
TOP_K_SIMILAR = 3
MAX_TESTS_PER_FUNCTION = 5

# MCP server
MCP_HOST = "127.0.0.1"
MCP_PORT = 8001

# FastAPI
API_HOST = "127.0.0.1"
API_PORT = 8000