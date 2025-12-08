from pathlib import Path
from typing import Tuple, List, Any
from rag_pipeline.config import ModelCfg
from rag_pipeline.indexing.sparse import build_sparse_retriever
from langchain_core.documents import Document

try:
    import chromadb  # type: ignore
    from langchain_chroma import Chroma  # type: ignore
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    chromadb = None
    Chroma = None
    HuggingFaceEmbeddings = None


def get_global_index() -> tuple[Any, Any, List]:
    """Load the global persisted Chroma index from `.chroma_rag` if present.

    Returns a tuple (vs, bm25, docs). When the index is not available,
    returns (None, None, []).
    """
    try:
        if chromadb is None or Chroma is None or HuggingFaceEmbeddings is None:
            return None, None, []

        persist_dir = ".chroma_rag"
        base = Path(persist_dir)
        if not base.exists():
            return None, None, []

        client = chromadb.PersistentClient(path=persist_dir)
        cols = client.list_collections()
        if not cols:
            return None, None, []

        col_name = cols[0].name
        embeddings = HuggingFaceEmbeddings(
            model_name=ModelCfg().embed_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        vs = Chroma(
            client=client,
            collection_name=col_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        try:
            raw = vs._collection.get(include=["documents","metadatas"])
            docs: List[Document] = []
            for doc, meta in zip(raw.get("documents",[]), raw.get("metadatas",[] )):
                if doc is None:
                    continue
                docs.append(Document(page_content=doc, metadata=meta or{}))
        except Exception:
            docs = []
        bm25 = build_sparse_retriever(docs) if docs else None
        
        return vs, bm25, docs
    except Exception:
        # Return empty on any failure; caller handles graceful empty results
        return None, None, []

