
import os
from pathlib import Path

# ===== File paths (flexible) =====
BASE_DIR = Path(__file__).resolve().parent
REVIEWS_FILE = Path(os.getenv("REVIEWS_FILE", BASE_DIR / "Software.jsonl"))
META_FILE = Path(os.getenv("META_FILE", BASE_DIR / "meta_Software.jsonl"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "output"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "50000"))
