"""
Configuration module for Smart Contract Assistant.
Loads settings from .env file and provides project-wide constants.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Project Root = This folder ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

# ── Fix Python Path ────────────────────────────────────────────────────────
# Ensure the parent of this package is on sys.path
# so 'from smart_contract_assistant import ...' works from anywhere.
_parent = PROJECT_ROOT.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

# ── Logging Setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smart_contract_assistant")

# ── Load Environment Variables ─────────────────────────────────────────────
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
    logger.debug(f"Loaded .env from: {_env_path}")

# ── Groq Settings (Chat Completions) ───────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Embeddings Settings (HuggingFace - Local/Free) ────────────────────────
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Ingestion Settings ─────────────────────────────────────────────────────
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ".", " "]

# ── Retriever Settings ─────────────────────────────────────────────────────
RETRIEVER_K = 4

# ── Paths (all relative to this folder) ────────────────────────────────────
FAISS_INDEX_PATH = str(PROJECT_ROOT / "faiss_index")

# ── Server Settings ────────────────────────────────────────────────────────
API_HOST = "localhost"
API_PORT = 8000
GRADIO_PORT = 7860

# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert legal assistant. Use the following pieces of "
    "retrieved context to answer the question. If you don't know the "
    "answer, say that you don't know. Keep the answer concise.\n\n"
    "{context}"
)


def validate_api_keys():
    """Validates that the Groq API key is configured."""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set (required for chat completions)\n\n"
            "Create a .env file from .env.example and add your Groq API key.\n"
            "Get your free key from: https://console.groq.com"
        )
    logger.info("API key configured: Groq (chat), HuggingFace (embeddings - local/free)")
