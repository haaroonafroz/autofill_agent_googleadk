"""
Central configuration for the autofill agent.

All settings are read from environment variables (or the project's .env file).
Model names live here so they can be changed without touching code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Main reasoning/computer-use model (fast, cheap, high volume).
AGENT_MODEL = _env("AUTOFILL_AGENT_MODEL", "gpt-6-luna")
# Escalation model for hard steps (used later by the agent loop).
HARD_MODEL = _env("AUTOFILL_HARD_MODEL", "gpt-6.1-terra")
# Reasoning effort for the agent model: none | low | medium | high | xhigh | max
AGENT_REASONING_EFFORT = _env("AUTOFILL_REASONING_EFFORT", "low")
# Embedding model for CV chunks.
EMBEDDING_MODEL = _env("AUTOFILL_EMBEDDING_MODEL", "text-embedding-3-small")
# Small model that labels each CV section at ingestion (kind, company, dates).
INDEX_MODEL = _env("AUTOFILL_INDEX_MODEL", "gpt-6-luna")
INDEX_REASONING_EFFORT = _env("AUTOFILL_INDEX_REASONING_EFFORT", "none")

# Local SQLite database: CV profiles, CV chunks + embeddings, answer memory,
# and LangGraph checkpoints.
DB_PATH = Path(_env("AUTOFILL_DB_PATH", str(PROJECT_ROOT / "data" / "autofill.db")))

# Default user when none is given on the command line.
DEFAULT_USER_ID = _env("AUTOFILL_USER_ID", "default")


def require_api_key() -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the .env file in the project root."
        )
    return OPENAI_API_KEY
