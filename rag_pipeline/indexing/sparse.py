# rag_pipeline/indexing/sparse.py
import re
import pickle
from langchain_community.retrievers import BM25Retriever
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

class BM25Wrapper:
    """BM25Retriever를 .invoke(query) 인터페이스로 감싸는 래퍼"""

    def __init__(self, retriever: BM25Retriever):
        self._r = retriever

    @property
    def k(self):
        return getattr(self._r, "k", None)

    @k.setter
    def k(self, value):
        setattr(self._r, "k", value)

    def invoke(self, query: str):
        # langchain 버전에 따라 invoke가 없을 수 있으니 안전하게 처리
        if hasattr(self._r, "invoke"):
            return self._r.invoke(query)
        return self._r.get_relevant_documents(query)




def build_sparse_retriever(docs, k: int = 8):
    """processing.py / pipeline 이 기대하는 팩토리 함수 (invoke 지원)"""
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return BM25Wrapper(bm25)  # 이 객체는 .invoke(query) 를 제공합니다.
