
from typing import List
from langchain_core.documents import Document
from .fusion import rrf_merge_score
from .rerankers import Reranker

def hybrid_retrieve(query: str, vs, bm25, k_dense: int, k_sparse: int, k_final: int, reranker: Reranker | None) -> List[Document]:
    dense_docs = vs.as_retriever(search_kwargs={"k": k_dense}).invoke(query)
    sparse_docs = bm25.invoke(query)
    ranked = rrf_merge_score(dense_docs, sparse_docs)
    if reranker:
        ranked_pairs = reranker.rerank(query, ranked[:max(k_dense, k_sparse)*2], top_k=k_final)
        return [d for d, _ in ranked_pairs]
    return ranked[:k_final]
