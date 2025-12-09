from typing import List, Dict, Any
from langchain_core.documents import Document
from .fusion import rrf_merge_score
from .rerankers import Reranker

def hybrid_retrieve(
    query: str, 
    vs, 
    bm25, 
    k_dense: int, 
    k_sparse: int, 
    k_final: int, 
    reranker: Reranker | None,
    filter: Dict[str, Any] | None = None  # 인자 추가됨
) -> List[Document]:
    
    # 🔥 [수정] 필터 적용 로직 추가
    dense_kwargs = {"k": k_dense}
    if filter:
        dense_kwargs["filter"] = filter  # 여기서 실제로 DB에 필터를 겁니다.
        
    # vs가 있을 때만 검색 (필터 포함된 옵션 전달)
    if vs:
        dense_docs = vs.as_retriever(search_kwargs=dense_kwargs).invoke(query)
    else:
        dense_docs = []
    
    # BM25는 필터 없이 전체 검색 (Sparse)
    sparse_docs = bm25.invoke(query) if bm25 else []
    
    # RRF로 결과 병합
    ranked = rrf_merge_score(dense_docs, sparse_docs)
    
    # 재랭킹 (Reranking)
    if reranker:
        ranked_pairs = reranker.rerank(query, ranked, top_k=k_final)
        return [d for d, _ in ranked_pairs]
    
    return ranked[:k_final]