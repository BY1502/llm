from __future__ import annotations

import time
import logging
import gc 
import torch # [추가] 메모리 정리용
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
    - [수정] 임베딩 모델을 CPU로 강제하여 GPU OOM 방지
    """
    # 1) Chroma client & collection
    client = chromadb.PersistentClient(path=str(idx_cfg.chroma_dir))
    coll = client.get_or_create_collection(name=idx_cfg.collection)

    # 2) Embedding 준비 (★ 여기가 핵심 수정 포인트)
    # GPU(cuda)가 아닌 CPU를 강제 사용하도록 설정합니다.
    embeddings = HuggingFaceEmbeddings(
        model_name=model_cfg.embed_model,
        model_kwargs={"device": "cpu"},  # [★수정] 무조건 CPU 사용!
        encode_kwargs={"normalize_embeddings": True},
    )

    if docs:
        BATCH = 32 # [★수정] 배치 크기를 조금 줄여서(128->32) 안정성 확보
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
                # [★추가] 루프 돌 때마다 메모리 청소 (안전장치)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                batch = tuples[s:s + BATCH]
                if not batch:
                    continue

                ids = [t[0] for t in batch]
                texts = [t[1] for t in batch]
                metas = [t[2] for t in batch]

                if not texts:
                    continue

                # 이제 CPU에서 임베딩을 수행하므로 GPU 메모리를 건드리지 않음
                vecs = embeddings.embed_documents(texts)
                
                if not vecs:
                    print(f"[VS] Embedding empty at offset {s}; skipping.")
                    continue

                coll.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=vecs)
                done += len(batch)

                if (s // BATCH) % 10 == 0:
                    elapsed = time.perf_counter() - t0
                    rps = done / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "Upsert progress: %d/%d (%.1f%%) | %.1f docs/s | %.1fs elapsed",
                        done, total, done / total * 100.0, rps, elapsed,
                    )

        elapsed = time.perf_counter() - t0
        logger.info(
            "Vectorstore upsert done: %d docs in %.2fs (%.1f docs/s)",
            done, elapsed, done / elapsed if elapsed > 0 else 0.0,
        )

    # 3) 검색용 래퍼 반환
    vs = Chroma(
        client=client,
        collection_name=idx_cfg.collection,
        embedding_function=embeddings,
    )
    return vs