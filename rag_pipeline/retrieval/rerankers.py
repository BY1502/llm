
from typing import Protocol, List, Tuple
from langchain_core.documents import Document

class Reranker(Protocol):
    def rerank(self, query: str, docs: List[Document], top_k: int) -> List[Tuple[Document, float]]: ...

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.ce = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, docs: List[Document], top_k: int = 5):
        pairs = [(query, d.page_content) for d in docs]
        scores = self.ce.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
