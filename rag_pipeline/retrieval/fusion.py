from langchain_core.documents import Document
from typing import List, Dict


def _key(d: Document):
    md = getattr(d, "metadata", {}) or {}
    return md.get("id") or md.get("source") or id(d)

def _rrf(docs: List[Document], k: int = 60) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for rank, d in enumerate(docs, start=1):
        key = _key(d)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return scores


def rrf_merge_score(dense_docs: List[Document], sparse_docs: List[Document]) -> List[Document]:
    s_dense = _rrf(dense_docs)
    s_sparse = _rrf(sparse_docs)
    
    pool = {}
    for d in dense_docs + sparse_docs:
        pool[_key(d)] = d
    combo = {k: s_dense.get(k, 0.0) + s_sparse.get(k, 0.0) for k in pool.keys()}
    return sorted(pool.values(), key=lambda d: combo.get(_key(d), 0.0), reverse=True)

