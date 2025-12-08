# rag_pipeline/utils/ids.py
from __future__ import annotations
import hashlib

def make_chunk_id(source: str, page: int | None, text: str) -> str:
    """같은 파일/페이지/텍스트(앞부분)면 항상 같은 ID가 나오도록 해시 생성."""
    head = (text or "").strip()[:200]
    key = f"{source}|{'' if page is None else page}|{head}"
    return hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()
