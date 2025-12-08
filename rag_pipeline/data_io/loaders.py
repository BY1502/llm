from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.language_models import BaseLanguageModel
from langchain_community.document_loaders import TextLoader, PyPDFLoader,PyMuPDFLoader

from rag_pipeline.config import ChunkCfg, PipelineCfg
from rag_pipeline.data_io.csv_schema import (
    FIELD_ALIASES,
    meta_get,
    make_csv_schema_report,
    print_csv_schema_report,
    save_csv_schema_report,
)
from rag_pipeline.data_io.readers import read_pdf, read_txt
from rag_pipeline.utils.text import extract_kv_metadata
from rag_pipeline.metadata.tag_extractor import extract_metadata_from_text

logger = logging.getLogger(__name__)

def _sanitize_metadata(meta: dict) -> dict:
    """
    Chroma에 넣기 전에 metadata 값을 primitive 타입으로 정리.
    - str, int, float, bool, None: 그대로 사용
    - list, tuple: ", "로 join해서 문자열로 변환
    - 그 외 타입: str()으로 변환
    """
    clean: dict = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, (list, tuple)):
            # tags 같은 경우: ['가설공사', '동바리', ...] → "가설공사, 동바리, ..."
            clean[k] = ", ".join(str(x) for x in v)
        else:
            clean[k] = str(v)
    return clean



def _try_read_csv(path):
    # 1) 파일 앞부분 샘플
    raw = Path(path).read_bytes()
    head = raw[:4096]

    # 2) 후보 인코딩/구분자
    encodings = ["cp949", "utf-8", "utf-8-sig", "euc-kr", "ISO-8859-1"]
    seps = [",", ";", "\t"]     # 콤마/세미콜론/탭

    # 3) pandas 유연 옵션
    common_kwargs = dict(
        engine="python",           # 더 관대함
        on_bad_lines="skip",       # 깨진 라인 건너뜀 (pandas>=1.5)
        dtype=str,                 # 모든 컬럼을 문자열로
        quoting=csv.QUOTE_MINIMAL, # 따옴표 처리
    )

    # 4) sep 자동 추정 1차 (csv.Sniffer)
    sniff_sep = None
    try:
        sample = head.decode("utf-8", errors="ignore")
        sniff = csv.Sniffer().sniff(sample, delimiters=";,\t")
        sniff_sep = sniff.delimiter
    except Exception:
        pass

    # 5) 시도 순서: (추정 sep + 각 인코딩) → (seps 전수 + 각 인코딩) → (sep=None)
    # 5-1) 추정 sep가 있으면 먼저 시도
    if sniff_sep:
        for enc in encodings:
            try:
                logger.debug("CSV sniff attempt: path=%s sep=%r enc=%s", path, sniff_sep, enc)
                return pd.read_csv(path, encoding=enc, sep=sniff_sep, **common_kwargs)
            except Exception as e:
                logger.debug(
                    "CSV sniff failed: path=%s sep=%r enc=%s error=%s",
                    path,
                    sniff_sep,
                    enc,
                    e,
                )

    # 5-2) 대표 구분자들 전수 시도
    for sep in seps:
        for enc in encodings:
            try:
                logger.debug("CSV attempt: path=%s sep=%r enc=%s", path, sep, enc)
                return pd.read_csv(path, encoding=enc, sep=sep, **common_kwargs)
            except Exception as e:
                logger.debug(
                    "CSV failed: path=%s sep=%r enc=%s error=%s",
                    path,
                    sep,
                    enc,
                    e,
                )
                # pass

    # 5-3) 마지막 시도: sep 자동(None) + 인코딩 전수
    for enc in encodings:
        try:
            logger.debug("CSV fallback attempt: path=%s sep=auto enc=%s", path, enc)
            return pd.read_csv(path, encoding=enc, sep=None, **common_kwargs)
        except Exception as e:
            logger.debug(
                "CSV fallback failed: path=%s sep=auto enc=%s error=%s",
                path,
                enc,
                e,
            )
            pass

    logger.error("CSV parsing failed for %s", path)
    return None

# def csv_rows_to_documents(path: Path, pipeline_cfg: PipelineCfg) -> List[Document]:
#     df = _try_read_csv(path)
#     if df is None or df.empty:
#         raise RuntimeError(f"Failed to read CSV (encoding/sep/lines): {path}")

#     df.columns = [str(c).strip() for c in df.columns]

#     report = make_csv_schema_report(df, path)
#     print_csv_schema_report(report)
#     save_csv_schema_report(
#         report,
#         str(pipeline_cfg.schema_report_dir) if pipeline_cfg.schema_report_dir else None,
#     )

#     # FIELD_ALIASES의 logical key 리스트를 기준으로, 실제 컬럼명은 meta_get이 알아서 매핑한다.
#     logical_keys = list(FIELD_ALIASES.keys())

#     docs: List[Document] = []
#     for i, (_, row) in enumerate(df.iterrows()):
#         # 1) 원본 row를 그대로 메타데이터에 넣는다 (파일에 있는 컬럼명 그대로).
#         md = {str(k).strip(): ("" if pd.isna(v) else str(v)) for k, v in row.items()}

#         # 2) 논리 키 기준으로 값 추출 (alias 포함).
#         body_lines: list[str] = []
#         normalized_meta: dict = {}
#         for key in logical_keys:
#             val = meta_get(md, key)
#             if val and val != "정보 없음":
#                 body_lines.append(f"{key}: {val}")
#                 # 메타데이터에도 표준화된 키로 한 번 더 저장해 둔다.
#                 normalized_meta[key] = val

#         body = "\n".join(body_lines).strip()

#         # 3) 최종 메타데이터: source + 원본 컬럼 + 표준화 키 + 고정 id
#         meta = {
#             "source": str(path),
#             **md,
#             **normalized_meta,
#         }
#         meta["id"] = f"{meta['source']}#p0#c{i}"

#         docs.append(
#             Document(
#                 page_content=body or str(md),
#                 metadata=meta,
#             )
#         )

#     return docs

# 모든 칼럼을 메타데이터로 추출
def csv_rows_to_documents(path: Path, pipeline_cfg: PipelineCfg) -> List[Document]:
    df = _try_read_csv(path)
    if df is None or df.empty:
        raise RuntimeError(f"Failed to read CSV (encoding/sep/lines): {path}")

    # 헤더 공백 정리
    df.columns = [str(c).strip() for c in df.columns]

    # (선택) 스키마 리포트는 그대로 유지
    report = make_csv_schema_report(df, path)
    print_csv_schema_report(report)
    save_csv_schema_report(
        report,
        str(pipeline_cfg.schema_report_dir) if pipeline_cfg.schema_report_dir else None,
    )

    docs: List[Document] = []

    for i, (_, row) in enumerate(df.iterrows()):
        # 1) 이 row의 메타데이터: CSV 컬럼명을 그대로 키로 사용
        md: dict[str, str] = {}
        for k, v in row.items():
            col_name = str(k).strip()
            # NaN/빈 문자열은 제외
            if pd.isna(v):
                continue
            val = str(v).strip()
            if not val:
                continue
            md[col_name] = val

        # 2) page_content = "컬럼명: 값" 줄로 쭉 이어붙이기
        body_lines: list[str] = [f"{key}: {val}" for key, val in md.items()]
        body = "\n".join(body_lines).strip()

        # 3) 최종 메타데이터: source + 원본 컬럼 그대로 + id
        meta = {
            "source": str(path),
            **md,
        }
        meta["id"] = f"{meta['source']}#p0#c{i}"

        docs.append(
            Document(
                page_content=body or str(md),
                metadata=meta,
            )
        )

    return docs


def load_documents_from_path(path: Path, chunk_cfg: ChunkCfg, pipeline_cfg: PipelineCfg,
                             llm_for_tags: BaseLanguageModel | None = None) -> List[Document]:
    if path.is_dir():
        docs: List[Document] = []
        for p in path.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".csv", ".txt", ".pdf"}:
                logger.info("Loading file for chunking: %s", p)
                docs.extend(load_documents_from_path(p, chunk_cfg, pipeline_cfg, llm_for_tags=llm_for_tags))
        return docs
    if path.suffix.lower() == ".csv":
        return csv_rows_to_documents(path, pipeline_cfg)
    elif path.suffix.lower() in {".txt", ".pdf"}:
        return generic_file_to_documents(path, chunk_cfg,llm_for_tags=llm_for_tags)
    return []


def generic_file_to_documents(
    path: Path,
    chunk_cfg: ChunkCfg,
    llm_for_tags: BaseLanguageModel | None = None,
) -> List[Document]:
    print(f"[CHUNK] generic_file_to_documents start: {path}")
    """
    TXT/PDF 등 일반 텍스트 파일을 청킹해서 LangChain Document 리스트로 변환.
    + (옵션) llm_for_tags 가 주어지면 문서 레벨 태그/메타데이터를 추출해서 각 청크 metadata에 포함.
    """

    # 1) 원본 텍스트 읽기 (확장자 기준)
    ext = path.suffix.lower()
    if ext == ".txt":
        raw = read_txt(path)
    elif ext == ".pdf":
        loader = PyMuPDFLoader(str(path))
        pages = loader.load()
        raw = "\n".join(p.page_content for p in pages)
        print(f"[PDF] PyMuPDFLoader loaded {len(pages)} pages from {path.name}")
    else:
        # raw = ""
        return [] # 추가

    if not raw or not raw.strip():
        return []

    # 2) 기본 메타데이터 (공통)
    base_meta: dict = {
        "source": str(path),
    }

    # 3) LLM으로 태그/메타데이터 추출
    if llm_for_tags is not None:
        try:
            tag_meta = extract_metadata_from_text(
                text=raw[:4000],
                filename=path.name,
                llm=llm_for_tags,
            )
        except Exception as e:
            print(f"[TAG] 메타데이터/태그 추출 실패: file={path} | err={e}")
            tag_meta = {}
        else:
            base_meta.update({
            "document_type": tag_meta.get("document_type"),
            "project_name": tag_meta.get("project_name"),
            "location": tag_meta.get("location"),
            "company": tag_meta.get("company"),
            "facility_type": tag_meta.get("facility_type"),
            "tags": tag_meta.get("tags", []),
            "source_filename": tag_meta.get("source_filename", path.name),
        })

    # 4) 키:값 메타데이터 추출
    kv_meta = extract_kv_metadata(raw)
    base_meta.update({f"meta:{k}": v for k, v in kv_meta.items()})

    # 5) 청킹
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.size,
        chunk_overlap=chunk_cfg.overlap,
        separators=["\n\n", "\n", ". ", ".", " "],
    )
    chunks = splitter.split_text(raw)

    print(
        f"[CHUNK] file={path.name} ext={path.suffix.lower()} "
        f"chunks={len(chunks)} size={chunk_cfg.size} overlap={chunk_cfg.overlap}"
    )

        # 6) Document 리스트 생성
    docs: List[Document] = []
    for i, ch in enumerate(chunks):
        raw_meta = {
            **base_meta,
            "chunk_id": i,
            "id": f"{base_meta['source']}#p0#c{i}",
        }

        # 🔥 Chroma가 먹을 수 있게 메타데이터 정리
        safe_meta = _sanitize_metadata(raw_meta)

        docs.append(
            Document(
                page_content=ch,
                metadata=safe_meta,
            )
        )

    return docs



