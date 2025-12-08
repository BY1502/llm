from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from rag_pipeline.metadata.tag_extractor import extract_metadata_from_file

logger = logging.getLogger(__name__)


def _load_pdf_as_docs(path: Path) -> List[Document]:
    """PDF를 LangChain Document 리스트로 로드."""
    loader = PyPDFLoader(str(path))
    return loader.load()


def _load_txt_as_docs(path: Path, encoding: str = "utf-8") -> List[Document]:
    """TXT를 LangChain Document 리스트로 로드."""
    loader = TextLoader(str(path), encoding=encoding)
    return loader.load()


def _chunk_docs(
    docs: Sequence[Document],
    chunk_size: int = 900,
    chunk_overlap: int = 100,
) -> List[Document]:
    """문서 리스트를 청킹."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_pdf_txt_dir(
    root_dir: str | Path,
    llm_for_tags: BaseLanguageModel,
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
    max_chars_for_tags: int = 4000,
) -> List[Document]:
    """
    PDF/TXT 파일들을 태그/메타데이터 + 청킹까지 해서
    최종적으로 LangChain Document 리스트로 반환.

    - root_dir: 탐색할 루트 디렉토리
    - llm_for_tags: tag_extractor에 넘길 LLM (태그/메타데이터 추출용)
    """
    root = Path(root_dir)
    all_chunks: List[Document] = []

    logger.info("[INGEST] PDF/TXT ingest 시작: root=%s", root)

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".txt"}:
            continue

        logger.info("[INGEST] 처리 대상 파일: %s", path)

        # 1) 문서 레벨 태그/메타데이터 추출
        try:
            meta = extract_metadata_from_file(
                path=path,
                llm=llm_for_tags,
                max_chars=max_chars_for_tags,
            )
        except Exception as e:
            logger.error("[INGEST] 메타데이터/태그 추출 실패: %s | file=%s", e, path)
            continue

        # 2) 원본 문서 로드
        if suffix == ".pdf":
            raw_docs = _load_pdf_as_docs(path)
        else:  # .txt
            raw_docs = _load_txt_as_docs(path)

        if not raw_docs:
            logger.warning("[INGEST] 로드된 문서가 비어있음: %s", path)
            continue

        # 3) 청킹
        chunks = _chunk_docs(
            raw_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # 4) 각 청크에 문서 레벨 메타데이터(태그 포함) 붙이기
        for c in chunks:
            # 원래 loader가 넣어준 metadata가 있으면 유지하면서 update
            c.metadata = c.metadata or {}
            c.metadata.update({
                "file_path": meta.get("file_path"),
                "document_type": meta.get("document_type"),
                "project_name": meta.get("project_name"),
                "location": meta.get("location"),
                "company": meta.get("company"),
                "facility_type": meta.get("facility_type"),
                "tags": meta.get("tags", []),
                "source_filename": meta.get("source_filename"),
            })

        all_chunks.extend(chunks)

    logger.info("[INGEST] 완료: 총 청크 수=%d", len(all_chunks))
    return all_chunks
