from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from rag_pipeline.config import ModelCfg, ChunkCfg, IndexCfg, PipelineCfg
from rag_pipeline.indexing.vectorstore import build_vectorstore
from rag_pipeline.data_io.loaders import load_documents_from_path
# CSV 쪽이 따로라면, 거기서 csv_rows_to_documents를 직접 import해서 써도 됨

def build_docs_for_global_index() -> List[Document]:
    """
    data_store 전체를 훑어서 전역 인덱스용 Document 리스트 생성.
    - CSV: 규칙 기반 태그 (이미 구현됨)
    - PDF/TXT: LLM 기반 메타/태그 삽입
    """
    base_folder = Path("data_store")
    chunk_cfg = ChunkCfg()
    pipeline_cfg = PipelineCfg()
    model_cfg = ModelCfg()

    print(f"[GLOBAL] scan base folder: {base_folder}")

    llm_for_tags = ChatOllama(model=model_cfg)

    # 🔥 정확한 호출 방식: pipeline_cfg 포함
    docs = load_documents_from_path(
        base_folder,
        chunk_cfg,
        pipeline_cfg,
        llm_for_tags=llm_for_tags,
    )

    print(f"[GLOBAL] total docs loaded={len(docs)}")
    return docs


def rebuild_global_index() -> None:
    """
    .chroma_rag 전역 인덱스를 LLM 태깅 포함해서 새로 생성.
    """
    model_cfg = ModelCfg()
    index_cfg = IndexCfg()

    persist_dir = Path(".chroma_rag")

    # 1) 기존 인덱스 삭제
    if persist_dir.exists():
        print(f"[GLOBAL] remove old index: {persist_dir}")
        shutil.rmtree(persist_dir)

    # 2) 전체 문서 생성
    docs = build_docs_for_global_index()
    if not docs:
        print("[GLOBAL] no docs, abort.")
        return

    # 3) 전역 인덱스 생성
    # index_cfg.chroma_dir = persist_dir
    # index_cfg.collection = "global"

    print(f"[GLOBAL] building vectorstore: docs={len(docs)} dir={persist_dir}")
    build_vectorstore(docs, model_cfg, index_cfg)

    print("[GLOBAL] rebuilt .chroma_rag with tags.")


if __name__ == "__main__":
    rebuild_global_index()
