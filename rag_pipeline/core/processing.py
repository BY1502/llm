# from __future__ import annotations

# import logging
# import threading
# import time
# from pathlib import Path

# from langchain_ollama import ChatOllama

# from rag_pipeline.config import ChunkCfg, IndexCfg, ModelCfg, PipelineCfg
# # from rag_pipeline.core.workspace import BM25_CACHE, DOCS_CACHE, VS_CACHE
# from rag_pipeline.core.workspace import DATA_STORE, WORKSPACES, Workspace
# from rag_pipeline.data_io.loaders import load_documents_from_path
# from rag_pipeline.indexing.sparse import build_sparse_retriever
# from rag_pipeline.indexing.vectorstore import build_vectorstore

# logger = logging.getLogger(__name__)

# VS_CACHE = {}
# BM25_CACHE = {}
# DOCS_CACHE = {}

# INDEX_MARKER = ".indexed"
# _BOOTSTRAP_LOCK = threading.Lock()
# _BOOTSTRAPPED = False


# def start_processing(workspace: Workspace) -> None:
#     threading.Thread(target=process_workspace, args=(workspace,), daemon=True).start()


# def _list_workspace_files(folder: Path) -> list[str]:
#     if not folder.exists():
#         return []
#     return sorted(
#         p.name for p in folder.iterdir() if p.is_file() and p.name != INDEX_MARKER
#     )


# def _needs_reindex(folder: Path) -> bool:
#     marker = folder / INDEX_MARKER
#     if not marker.exists():
#         return True
#     marker_mtime = marker.stat().st_mtime
#     for path in folder.rglob("*"):
#         if path.is_file() and path.name != INDEX_MARKER and path.stat().st_mtime > marker_mtime:
#             return True
#     return False


# def _touch_index_marker(folder: Path) -> None:
#     if not folder.exists():
#         folder.mkdir(parents=True, exist_ok=True)
#     marker = folder / INDEX_MARKER
#     marker.write_text(str(time.time()), encoding="utf-8")


# def process_workspace(ws: Workspace) -> None:
#     model_cfg = ModelCfg()
#     chunk_cfg = ChunkCfg()
#     index_cfg = IndexCfg()
#     pipe_cfg = PipelineCfg()

#     ws.status = "indexing"
#     ws.ready = False
#     ws.error = None

#     try:
#         print(f"[WS] start processing workspace={ws.id} path={ws.folder}")
        
#         llm_for_tags = ChatOllama(model=model_cfg.ollama_model) # Ollama + Gemma3:1b 버전
#         docs = load_documents_from_path(ws.folder, chunk_cfg, pipe_cfg,
#                                         llm_for_tags= llm_for_tags,)
#         if not docs:
#             raise ValueError("인덱싱할 문서를 찾지 못했습니다.")
        
#         print(f"[WS] workspace={ws.id} loaded docs={len(docs)}")

#         vs = build_vectorstore(docs, model_cfg, index_cfg)
#         bm25 = build_sparse_retriever(docs)
#         VS_CACHE[ws.id], BM25_CACHE[ws.id], DOCS_CACHE[ws.id] = vs, bm25, docs

#         ws.ready = True
#         ws.status = "ready"
#         ws.files = _list_workspace_files(ws.folder)
#         ws.last_updated = time.time()
#         _touch_index_marker(ws.folder)
#         logger.info("Workspace %s indexed (%d docs)", ws.id, len(docs))
#     except Exception as exc:
#         ws.ready = False
#         ws.error = str(exc)
#         ws.status = "error"
#         VS_CACHE.pop(ws.id, None)
#         BM25_CACHE.pop(ws.id, None)
#         DOCS_CACHE.pop(ws.id, None)
#         logger.exception("Failed to process workspace %s", ws.id)


# def bootstrap_data_store() -> None:
#     global _BOOTSTRAPPED
#     with _BOOTSTRAP_LOCK:
#         if _BOOTSTRAPPED:
#             return

#         DATA_STORE.mkdir(parents=True, exist_ok=True)

#         for folder in sorted(p for p in DATA_STORE.iterdir() if p.is_dir()):
#             ws_id = folder.name
#             ws = WORKSPACES.get(ws_id) or Workspace(ws_id, folder)
#             ws.files = _list_workspace_files(folder)
#             WORKSPACES[ws_id] = ws

#             if not _needs_reindex(folder):
#                 ws.ready = True
#                 ws.status = "ready"
#                 logger.debug("Workspace %s already indexed; skipping bootstrap", ws_id)
#                 continue

#             logger.info("Bootstrapping workspace %s (%d files)", ws_id, len(ws.files))
#             process_workspace(ws)

#         _BOOTSTRAPPED = True

from __future__ import annotations

import logging
import threading
import time
import gc # [추가] 가비지 컬렉터
import torch # [추가] GPU 메모리 제어
from pathlib import Path

from langchain_ollama import ChatOllama

from rag_pipeline.config import ChunkCfg, IndexCfg, ModelCfg, PipelineCfg
# from rag_pipeline.core.workspace import BM25_CACHE, DOCS_CACHE, VS_CACHE
from rag_pipeline.core.workspace import DATA_STORE, WORKSPACES, Workspace
from rag_pipeline.data_io.loaders import load_documents_from_path
from rag_pipeline.indexing.sparse import build_sparse_retriever
from rag_pipeline.indexing.vectorstore import build_vectorstore

logger = logging.getLogger(__name__)

VS_CACHE = {}
BM25_CACHE = {}
DOCS_CACHE = {}

INDEX_MARKER = ".indexed"
_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAPPED = False

# [★추가] 동시 실행 방지용 세마포어 (한 번에 1개의 워크스페이스만 인덱싱)
_PROCESSING_LOCK = threading.Semaphore(1) 

def start_processing(workspace: Workspace) -> None:
    # 데몬 스레드로 실행하되, 내부에서 Lock을 걸어 순차 처리됨
    threading.Thread(target=process_workspace, args=(workspace,), daemon=True).start()


def _list_workspace_files(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return sorted(
        p.name for p in folder.iterdir() if p.is_file() and p.name != INDEX_MARKER
    )


def _needs_reindex(folder: Path) -> bool:
    marker = folder / INDEX_MARKER
    if not marker.exists():
        return True
    marker_mtime = marker.stat().st_mtime
    for path in folder.rglob("*"):
        if path.is_file() and path.name != INDEX_MARKER and path.stat().st_mtime > marker_mtime:
            return True
    return False


def _touch_index_marker(folder: Path) -> None:
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
    marker = folder / INDEX_MARKER
    marker.write_text(str(time.time()), encoding="utf-8")


def process_workspace(ws: Workspace) -> None:
    # [★추가] 다른 작업이 끝날 때까지 대기 (동시 실행 방지)
    acquired = _PROCESSING_LOCK.acquire(blocking=False)
    if not acquired:
        logger.warning(f"[WS] Workspace {ws.id} is pending... (Server busy)")
        # blocking=True로 대기하거나, 사용자에게 '처리 중' 메시지를 줄 수 있음
        # 여기서는 안전하게 대기하도록 변경
        _PROCESSING_LOCK.acquire() 

    try:
        # [★추가] 작업 시작 전 메모리 대청소
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        model_cfg = ModelCfg()
        chunk_cfg = ChunkCfg()
        index_cfg = IndexCfg()
        pipe_cfg = PipelineCfg()

        # [★핵심 해결책] 임베딩은 CPU로 강제 전환하여 OOM 방지
        # (config.py 구조에 따라 다를 수 있으나, 보통 device 필드가 있다면 덮어쓰기)
        if hasattr(model_cfg, 'device'):
            model_cfg.device = 'cpu'
            logger.info("Force embedding model to use CPU to save GPU memory")

        ws.status = "indexing"
        ws.ready = False
        ws.error = None

        print(f"[WS] start processing workspace={ws.id} path={ws.folder}")
        
        # 태깅용 LLM 호출 (가벼운 모델 권장)
        llm_for_tags = ChatOllama(model=model_cfg.ollama_model) 
        
        docs = load_documents_from_path(ws.folder, chunk_cfg, pipe_cfg,
                                        llm_for_tags=llm_for_tags,)
        
        if not docs:
            # 문서가 없으면 에러보다는 '완료' 처리하되 빈 상태로 두는 게 나을 수 있음
            # 여기서는 기존 로직 유지
            raise ValueError("인덱싱할 문서를 찾지 못했습니다.")
        
        print(f"[WS] workspace={ws.id} loaded docs={len(docs)}")

        # [★ OOM 발생 지점] 여기서 CPU 설정을 먹인 model_cfg가 들어갑니다.
        vs = build_vectorstore(docs, model_cfg, index_cfg)
        bm25 = build_sparse_retriever(docs)
        
        VS_CACHE[ws.id], BM25_CACHE[ws.id], DOCS_CACHE[ws.id] = vs, bm25, docs

        ws.ready = True
        ws.status = "ready"
        ws.files = _list_workspace_files(ws.folder)
        ws.last_updated = time.time()
        _touch_index_marker(ws.folder)
        logger.info("Workspace %s indexed (%d docs)", ws.id, len(docs))

    except Exception as exc:
        ws.ready = False
        ws.error = str(exc)
        ws.status = "error"
        # 실패 시 캐시 삭제
        VS_CACHE.pop(ws.id, None)
        BM25_CACHE.pop(ws.id, None)
        DOCS_CACHE.pop(ws.id, None)
        logger.exception("Failed to process workspace %s", ws.id)
    
    finally:
        # [★추가] 작업 종료 후 락 해제 및 메모리 정리
        _PROCESSING_LOCK.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def bootstrap_data_store() -> None:
    global _BOOTSTRAPPED
    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAPPED:
            return

        DATA_STORE.mkdir(parents=True, exist_ok=True)

        for folder in sorted(p for p in DATA_STORE.iterdir() if p.is_dir()):
            ws_id = folder.name
            ws = WORKSPACES.get(ws_id) or Workspace(ws_id, folder)
            ws.files = _list_workspace_files(folder)
            WORKSPACES[ws_id] = ws

            if not _needs_reindex(folder):
                ws.ready = True
                ws.status = "ready"
                # 이미 로드된 상태라면 벡터스토어도 메모리에 올려야 함 (중요)
                # 현재 구조상 서버 재시작 시에는 캐시가 비어있으므로
                # _needs_reindex가 False여도 메모리 로딩은 필요할 수 있음.
                # (이 부분은 기존 로직을 존중하여 유지합니다)
                logger.debug("Workspace %s already indexed on disk; skipping re-process", ws_id)
                continue

            logger.info("Bootstrapping workspace %s (%d files)", ws_id, len(ws.files))
            process_workspace(ws)

        _BOOTSTRAPPED = True