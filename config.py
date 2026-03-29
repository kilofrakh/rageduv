# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL         = "openai/gpt-oss-120b"      # Groq model

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"       # local, no API key needed

# ── Vector store ──────────────────────────────────────────────────────────────
CHROMA_DIR      = "./chroma_db"            # persisted on disk
COLLECTION_NAME = "course_materials"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512                        # tokens per chunk
CHUNK_OVERLAP = 64                         # overlap between chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5                                  # number of chunks retrieved per query

# ── Analytics DB ──────────────────────────────────────────────────────────────
ANALYTICS_DB = "sqlite:///analytics.db"

# ── Upload directory ──────────────────────────────────────────────────────────
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
