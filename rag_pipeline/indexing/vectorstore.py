# from __future__ import annotations
# import time
# import logging
# from typing import List

# import chromadb
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document

# from ..config import ModelCfg, IndexCfg
# from ..utils.ids import make_chunk_id

# logger = logging.getLogger(__name__)

# def build_vectorstore(docs: List[Document], model_cfg: ModelCfg, idx_cfg: IndexCfg) -> Chroma:
#     """
#     기존 add_documents -> Chroma 네이티브 upsert 로 변경.
#     - 같은 id면 갱신, 없으면 삽입 (중복 방지)
#     - upsert 이후, 검색을 위해 LangChain Chroma 래퍼를 반환
#     """
#     # 1) Chroma client & collection
#     client = chromadb.PersistentClient(path=str(idx_cfg.chroma_dir))
#     coll = client.get_or_create_collection(name=idx_cfg.collection)  # <-- upsert 지원

#     # 2) Embedding 준비 (직접 계산해서 upsert에 embeddings로 전달)
#     embeddings = HuggingFaceEmbeddings(
#         model_name=model_cfg.embed_model,
#         encode_kwargs={"normalize_embeddings": True}
#     )

#     if docs:
#         BATCH = 128
#         total = len(docs)
#         done = 0
#         t0 = time.perf_counter()

#         logger.info(
#             "Vectorstore upsert start: docs=%d batch=%d collection=%s dir=%s",
#             total, BATCH, idx_cfg.collection, idx_cfg.chroma_dir
#         )

#         # Document -> (id, text, metadata) 변환
#         def _to_tuple(d: Document):
#             meta = dict(getattr(d, "metadata", {}) or {})
#             source = meta.get("source") or meta.get("doc_id") or "unknown"
#             page = meta.get("page")
#             text = d.page_content or ""
#             _id = make_chunk_id(str(source), page, text)
#             # 디버깅 편의를 위해 메타에도 심어두기
#             meta.setdefault("chunk_id", _id)
#             return _id, text, meta

#         seen_ids = set()
#         tuples = []
#         for d in docs:
#             _id, text, meta = _to_tuple(d)
#             if _id in seen_ids:
#                 # 필요하면 logger 로 찍어도 됨
#                 # logger.warning("Duplicate chunk id detected, skip: %s (source=%s)", _id, meta.get("source"))
#                 continue
#             seen_ids.add(_id)
#             tuples.append((_id, text, meta))

#         for s in range(0, total, BATCH):
#             batch = tuples[s:s+BATCH]
#             ids = [t[0] for t in batch]
#             texts = [t[1] for t in batch]
#             metas = [t[2] for t in batch]
#             logger.debug("Embedding batch size: %d", len(texts))

#             vecs = embeddings.embed_documents(texts)  # 임베딩 직접 계산
#             if not vecs:
#                 logger.error("Embedding funtion returned an empty list for texts batch size:%d",len(texts))
#             # 🔁 핵심: add() 아님. upsert() 사용
#             coll.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vecs)

#             done += len(batch)
#             # 가벼운 진행 로그
#             if (s // BATCH) % 10 == 0:
#                 elapsed = time.perf_counter() - t0
#                 rps = done / elapsed if elapsed > 0 else 0.0
#                 logger.info("Upsert progress: %d/%d (%.1f%%) | %.1f docs/s | %.1fs elapsed",
#                             done, total, done/total*100.0, rps, elapsed)

#         elapsed = time.perf_counter() - t0
#         logger.info("Vectorstore upsert done: %d docs in %.2fs (%.1f docs/s)",
#                     done, elapsed, done/elapsed if elapsed > 0 else 0.0)

#     # 3) 검색용 래퍼 반환 (이후 similarity_search 등 그대로 사용)
#     vs = Chroma(
#         client=client,
#         collection_name=idx_cfg.collection,
#         embedding_function=embeddings
#     )
#     return vs



from __future__ import annotations

import time
import logging
from typing import List

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..config import ModelCfg, IndexCfg
from ..utils.ids import make_chunk_id

logger = logging.getLogger(__name__)


def build_vectorstore(docs: List[Document], model_cfg: ModelCfg, idx_cfg: IndexCfg) -> Chroma:
    """
    - LangChain Chroma 래퍼 + Chroma Native upsert 사용
    - chunk_id 기준으로 중복 제거
    - 빈 텍스트/빈 배치 방어
    """
    # 1) Chroma client & collection
    client = chromadb.PersistentClient(path=str(idx_cfg.chroma_dir))
    coll = client.get_or_create_collection(name=idx_cfg.collection)

    # 2) Embedding 준비
    embeddings = HuggingFaceEmbeddings(
        model_name=model_cfg.embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    if docs:
        BATCH = 128
        t0 = time.perf_counter()

        logger.info(
            "Vectorstore upsert start: docs=%d batch=%d collection=%s dir=%s",
            len(docs),
            BATCH,
            idx_cfg.collection,
            idx_cfg.chroma_dir,
        )

        # --- Document -> (id, text, metadata) 변환 + 중복/빈텍스트 제거 ---
        def _to_tuple(d: Document):
            meta = dict(getattr(d, "metadata", {}) or {})
            source = meta.get("source") or meta.get("doc_id") or "unknown"
            page = meta.get("page")
            text = (d.page_content or "").strip()

            if not text:
                # 내용이 완전히 비어있으면 None 반환해서 스킵
                return None

            _id = make_chunk_id(str(source), page, text)
            meta.setdefault("chunk_id", _id)
            return _id, text, meta

        seen_ids = set()
        tuples: List[tuple[str, str, dict]] = []

        for d in docs:
            t = _to_tuple(d)
            if t is None:
                continue
            _id, text, meta = t
            if _id in seen_ids:
                # logger.debug("Duplicate chunk id skip: %s", _id)
                continue
            seen_ids.add(_id)
            tuples.append((_id, text, meta))

        total = len(tuples)
        print(f"[VS] unique chunks={total} (from docs={len(docs)})")

        done = 0
        if total == 0:
            logger.warning("No unique non-empty chunks to upsert; skipping Chroma upsert.")
        else:
            for s in range(0, total, BATCH):
                batch = tuples[s:s + BATCH]
                if not batch:
                    # 빈 배치는 바로 스킵 (중복 제거 이후 tail에서 나올 수 있음)
                    continue

                ids = [t[0] for t in batch]
                texts = [t[1] for t in batch]
                metas = [t[2] for t in batch]

                # 혹시 모를 빈 텍스트 배치 방어
                if not texts:
                    print(f"[VS] skip batch at offset {s} (empty texts)")
                    continue

                vecs = embeddings.embed_documents(texts)
                if not vecs:
                    print(
                        f"[VS] Embedding function returned empty list at offset {s}, "
                        f"texts_len={len(texts)}; skipping this batch."
                    )
                    continue

                coll.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vecs)
                done += len(batch)

                if (s // BATCH) % 10 == 0:
                    elapsed = time.perf_counter() - t0
                    rps = done / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "Upsert progress: %d/%d (%.1f%%) | %.1f docs/s | %.1fs elapsed",
                        done,
                        total,
                        done / total * 100.0,
                        rps,
                        elapsed,
                    )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Vectorstore upsert done: %d docs in %.2fs (%.1f docs/s)",
            done,
            elapsed,
            done / elapsed if elapsed > 0 else 0.0,
        )

    # 3) 검색용 래퍼 반환
    vs = Chroma(
        client=client,
        collection_name=idx_cfg.collection,
        embedding_function=embeddings,
    )
    return vs
