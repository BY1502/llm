from __future__ import annotations

import logging
import threading
import time
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


def start_processing(workspace: Workspace) -> None:
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
    model_cfg = ModelCfg()
    chunk_cfg = ChunkCfg()
    index_cfg = IndexCfg()
    pipe_cfg = PipelineCfg()

    ws.status = "indexing"
    ws.ready = False
    ws.error = None

    try:
        print(f"[WS] start processing workspace={ws.id} path={ws.folder}")
        
        llm_for_tags = ChatOllama(model=model_cfg.ollama_model) # Ollama + Gemma3:1b 버전
        docs = load_documents_from_path(ws.folder, chunk_cfg, pipe_cfg,
                                        llm_for_tags= llm_for_tags,)
        if not docs:
            raise ValueError("인덱싱할 문서를 찾지 못했습니다.")
        
        print(f"[WS] workspace={ws.id} loaded docs={len(docs)}")

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
        VS_CACHE.pop(ws.id, None)
        BM25_CACHE.pop(ws.id, None)
        DOCS_CACHE.pop(ws.id, None)
        logger.exception("Failed to process workspace %s", ws.id)


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
                logger.debug("Workspace %s already indexed; skipping bootstrap", ws_id)
                continue

            logger.info("Bootstrapping workspace %s (%d files)", ws_id, len(ws.files))
            process_workspace(ws)

        _BOOTSTRAPPED = True
